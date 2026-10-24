"""FeatTracker: short- and long-range temporal feature consistency for UOT."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.layers.lrtfc import LRTFC
from lib.models.layers.srtfc import DFE, SRTFC
from lib.models.feattracker.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.feattracker.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class FeatTracker(nn.Module):
    """FeatTracker model.

    The visual encoder and the tracking head follow ODTrack. Two feature-level
    consistency branches are inserted between the encoder and the head:

    * the short-range branch (``DFE`` + ``SRTFC``) refines the current feature
      map and aligns it with a memory of recent features;
    * the long-range branch (``LRTFC``) propagates the temporal token through
      a wavelet-based decomposition.

    Args:
        transformer: the ViT backbone.
        box_head: the center head.
        aux_loss: whether to use auxiliary losses.
        head_type: ``CORNER`` or ``CENTER``.
        token_len: number of temporal tokens propagated across frames.
        cfg: the experiment configuration.
    """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CENTER", token_len=1, cfg=None):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type

        self.use_dfe = bool(getattr(cfg.MODEL.DFE, 'ENABLED', True))
        self.use_srtfc = bool(getattr(cfg.MODEL.SRTFC, 'ENABLED', True))
        self.use_lrtfc = bool(getattr(cfg.MODEL.LRTFC, 'ENABLED', True))

        self.dfe = None
        if self.use_dfe:
            self.dfe = DFE(
                dim=transformer.embed_dim,
                cond_dim=getattr(cfg.MODEL.DFE, 'COND_DIM', 128),
                num_steps=getattr(cfg.MODEL.DFE, 'NUM_STEPS', 700),
                inference_step=getattr(cfg.MODEL.DFE, 'INFERENCE_STEP', 12),
                stochastic_inference=getattr(cfg.MODEL.DFE, 'STOCHASTIC_INFERENCE', True),
            )

        self.srtfc = None
        if self.use_srtfc:
            self.srtfc = SRTFC(
                dim=transformer.embed_dim,
                memory_size=getattr(cfg.MODEL.SRTFC, 'MEMORY_SIZE', 5),
                strategy=getattr(cfg.MODEL.SRTFC, 'MEMORY_STRATEGY', 'default'),
            )

        self.lrtfc = None
        if self.use_lrtfc:
            self.lrtfc = LRTFC(
                dim=transformer.embed_dim,
                separation=getattr(cfg.MODEL.LRTFC, 'SEPARATION', 'haar'),
            )

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.dim = transformer.embed_dim
        self.token_len = token_len

        self.track_query = None
        self.track_query_prev = None

    def apply_short_range(self, x):
        """Apply DFE and the short-range memory pool to encoder features."""
        if isinstance(x, list):
            originals = [feat.clone().detach() for feat in x]
            enhanced = [self.dfe(feat) for feat in x] if self.dfe is not None else list(x)
            if self.srtfc is None:
                return enhanced
            return [self.srtfc(o, e) for o, e in zip(originals, enhanced)]

        original = x.clone().detach()
        enhanced = self.dfe(x) if self.dfe is not None else x
        if self.srtfc is None:
            return enhanced
        return self.srtfc(original, enhanced)

    def update_track_query(self, x):
        """Propagate the temporal token of the current frame."""
        if not self.backbone.add_cls_token:
            return

        current_tokens = x[:, :self.token_len].clone().detach()

        if self.lrtfc is None:
            self.track_query = current_tokens
            return

        if self.track_query_prev is not None:
            self.track_query_prev = self.track_query_prev.to(current_tokens.device)
            fused = self.lrtfc(self.track_query_prev, current_tokens)
            self.track_query = fused[:, :self.token_len, :].detach()
        else:
            # The first frame has no history: pad the token to the minimum
            # length expected by the wavelet decomposition.
            self.track_query = F.pad(
                current_tokens,
                (0, 0, 0, max(0, 4 - current_tokens.size(1))),
            )[:, :self.token_len, :].detach()

        self.track_query_prev = F.pad(
            self.track_query.clone().detach(),
            (0, 0, 0, max(0, 4 - self.track_query.size(1))),
            value=0,
        )

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search must be a list."

        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone(
                z=template.copy(),
                x=search[i],
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
                track_query=self.track_query,
                token_len=self.token_len,
            )

            if self.dfe is not None or self.srtfc is not None:
                x = self.apply_short_range(x)

            feat_last = x[-1] if isinstance(x, list) else x
            enc_opt = feat_last[:, -self.feat_len_s:]

            self.update_track_query(x)

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                (0, 3, 2, 1)).contiguous()

            out = self.forward_head(opt, None)
            out.update(aux_dict)
            out['backbone_feat'] = x
            out_dict.append(out)

        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """Run the tracking head on the attention-weighted search features."""
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            return {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map,
            }

        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord_new = bbox.view(bs, Nq, 4)
            return {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map,
            }

        raise NotImplementedError

    def reset(self):
        """Reset the temporal state between sequences."""
        self.track_query = None
        self.track_query_prev = None
        if self.srtfc is not None:
            self.srtfc.reset()


def build_feattracker(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
            attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,
        )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
            attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,
        )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
        )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
        )

    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = FeatTracker(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
        cfg=cfg,
    )

    return model

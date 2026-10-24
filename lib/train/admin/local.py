import os

# Project root: <root>/lib/train/admin/local.py -> <root>
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DATA = os.path.join(_ROOT, 'data')


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = _ROOT    # Base directory for checkpoints and logs.
        self.tensorboard_dir = os.path.join(_ROOT, 'tensorboard')
        self.pretrained_networks = os.path.join(_ROOT, 'pretrained_networks')

        # Generic tracking datasets
        self.lasot_dir = os.path.join(_DATA, 'lasot')
        self.got10k_dir = os.path.join(_DATA, 'got10k/train')
        self.got10k_val_dir = os.path.join(_DATA, 'got10k/val')
        self.lasot_lmdb_dir = os.path.join(_DATA, 'lasot_lmdb')
        self.got10k_lmdb_dir = os.path.join(_DATA, 'got10k_lmdb')
        self.trackingnet_dir = os.path.join(_DATA, 'trackingnet')
        self.trackingnet_lmdb_dir = os.path.join(_DATA, 'trackingnet_lmdb')
        self.coco_dir = os.path.join(_DATA, 'coco')
        self.coco_lmdb_dir = os.path.join(_DATA, 'coco_lmdb')
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = os.path.join(_DATA, 'vid')
        self.imagenet_lmdb_dir = os.path.join(_DATA, 'vid_lmdb')
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''

        # Underwater tracking datasets
        self.webuot_dir = os.path.join(_DATA, 'webuot')
        self.uot100_dir = os.path.join(_DATA, 'uot100')
        self.uvot400_dir = os.path.join(_DATA, 'uvot400/train')
        self.uot32_dir = os.path.join(_DATA, 'uot32')
        self.utb180_dir = os.path.join(_DATA, 'utb180')
        self.ruod_dir = os.path.join(_DATA, 'ruod')
        self.vmat_dir = os.path.join(_DATA, 'vmat/jpgs')
        self.HUT290_dir = os.path.join(_DATA, 'hut290')

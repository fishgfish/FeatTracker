import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class VMAT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):

        root = env_settings().vmat_dir if root is None else root

        super().__init__('VMAT', root, image_loader)

        self.sequence_list = self._build_sequence_list(split)

    def get_name(self):
        return 'VMAT'

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _build_sequence_list(self, split=None):
        try:
            file_path = self.root
            sequence_list = os.listdir(file_path)
            random.shuffle(sequence_list)
        except:
            if split is not None:
                ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
                if split == 'train':
                    file_path = os.path.join(self.root, 'Train')
                elif split == 'test':
                    file_path = os.path.join(self.root, 'Test')
                else:
                    raise ValueError('Unknown split name.')
                sequence_list = os.listdir(file_path)
                random.shuffle(sequence_list)
            else:
                raise ValueError('Set split_name.')

        return sequence_list

    def _read_bb_anno(self, seq_path):
        seq_path=seq_path.replace('jpgs', 'annotations')
        bb_anno_file = seq_path + '_obj0.txt'
        # bb_anno_file = os.path.join(seq_path, "obj.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    ## Underwater Images
    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    ## Clear Images
    def _get_clear_img_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_grountgruth_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_sequence_info(self, seq_id):
        bbox_path = self._get_grountgruth_path(seq_id)
        bbox = self._read_bb_anno(bbox_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        # return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
        return {'bbox': bbox, 'valid': valid, 'visible': visible, }

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:05}.jpg'.format(frame_id+1))
        # if os.path.exists(os.path.join(seq_path, 'imgs/{:08}.jpg'.format(frame_id))):
        #     return os.path.join(seq_path, 'imgs/{:08}.jpg'.format(frame_id))    # frames start from 0
        # else:
        #     return os.path.join(seq_path, 'imgs/{:08}.png'.format(frame_id))    # some image is png

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_event_sequence_path(self, seq_id):        ## get events' frames
        return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id] + "_voxel")

    ###############################################################
    def _read_language(self, seq_path):
        try:
            language_file = os.path.join(seq_path, "language.txt")
            with open(language_file, 'r') as file:
                language = file.readline().replace(".", "").replace("\n", "").lower()
            return language
        except:
            return None
    ###############################################################

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

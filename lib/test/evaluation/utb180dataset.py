import os
import json
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from ..utils.load_text import load_text


class UTB180Dataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.utb180_path
        self.sequence_list = self._get_sequence_list(self.base_path)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64)
        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'imgs')

        num_images = len([f for f in os.listdir(frames_path) if f.endswith('.jpg')])  # Number of frames in this sequence.

        if num_images >= 1000:
            frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in
                           range(1, ground_truth_rect.shape[0] + 1)]
        elif num_images >= 100:
            frames_list = ['{}/{:04d}.jpg'.format(frames_path, frame_number) for frame_number in
                           range(1, ground_truth_rect.shape[0] + 1)]
        else:
            frames_list = ['{}/{:03d}.jpg'.format(frames_path, frame_number) for frame_number in
                           range(1, ground_truth_rect.shape[0] + 1)]

        # frame_event_list = []
        # event_img_list = []

        return Sequence(sequence_name, frames_list, 'utb180', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_list = os.listdir(self.base_path)
        return sequence_list

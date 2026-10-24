import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class WebUOTDataset(BaseDataset):

    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.webuot_path,'Test')
        else:
            self.base_path = os.path.join(self.env_settings.webuot_path, 'Train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        # ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'imgs')

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        # frame_event_list = []
        # event_img_list = []

        return Sequence(sequence_name, frames_list, 'webuot', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_list = os.listdir(self.base_path)
        return sequence_list

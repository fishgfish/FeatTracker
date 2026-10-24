import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VMATDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = os.path.join(self.env_settings.vmat_path, 'jpgs')
        self.sequence_list = self._get_sequence_list(self.base_path)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = self.base_path.replace('jpgs', 'annotations')
        anno_name = sequence_name + '_obj0.txt'
        anno_path = '{}/{}'.format(seq_path, anno_name)

        ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64)
        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        # frame_event_list = []
        # event_img_list = []

        return Sequence(sequence_name, frames_list, 'vmat', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_list = os.listdir(self.base_path)
        return sequence_list

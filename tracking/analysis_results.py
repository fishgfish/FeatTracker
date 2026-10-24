"""Print benchmark scores from cached tracking results.

Run ``tracking/test.py`` first so that the result files exist under
``output/test/tracking_results/feattracker/<param>_<runid>/``.
"""

import argparse
import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import matplotlib.pyplot as plt

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist

plt.rcParams['figure.figsize'] = [8, 8]


def parse_args():
    parser = argparse.ArgumentParser(description='Report tracking results.')
    parser.add_argument('--param', type=str, default='feattracker', help='Parameter file name.')
    parser.add_argument('--runid', type=int, default=300,
                        help='Checkpoint epoch used for the run. Defaults to 300.')
    parser.add_argument('--datasets', nargs='+', default=[
        'webuot', 'uwcot220', 'utb180', 'vmat', 'hut290', 'uot100', 'watb',
    ], help='Dataset names.')
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset_name in args.datasets:
        trackers = trackerlist(
            name='feattracker',
            parameter_name=args.param,
            dataset_name=dataset_name,
            run_ids=args.runid,
            display_name='FeatTracker',
        )

        dataset = get_dataset(dataset_name)
        print_results(
            trackers,
            dataset,
            dataset_name,
            merge_results=True,
            plot_types=('success', 'norm_prec', 'prec'),
        )


if __name__ == '__main__':
    main()

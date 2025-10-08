import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.plot_results import print_results

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                checkpoint=None, num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    seed = 3407
    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, checkpoint)]

    run_dataset(dataset, trackers, seed, debug, threads, num_gpus=num_gpus)

    report_text = print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
    import shutil
    shutil.rmtree(os.path.join(prj_path, 'output/test'))
    with open(os.path.join(prj_path, dataset_name + '_log.txt'), "a") as f:
        f.write(report_text + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--ckpt', type=str, default=None, help='The checkpoint path.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, args.ckpt, num_gpus=args.num_gpus)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    main()

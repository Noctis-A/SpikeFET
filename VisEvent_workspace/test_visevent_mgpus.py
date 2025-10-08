import os
import cv2
import sys
from os.path import join, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.spikefet import SpikeFET
import lib.test.parameter.spikefet as parameters
import multiprocessing
import torch
import time
import random

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _read_image(image_file: str):
    im = cv2.imread(image_file)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def genConfig(seq_path, set_type):
    if set_type == 'VisEvent':
        img_list = sorted([seq_path + '/vis_imgs/' + p for p in os.listdir(seq_path + '/vis_imgs') if os.path.splitext(p)[1] == '.bmp'])
        E_img_list = sorted([seq_path + '/event_imgs/' + p for p in os.listdir(seq_path + '/event_imgs') if os.path.splitext(p)[1] == '.bmp'])

        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        absent_label = np.loadtxt(seq_path + '/absent_label.txt')

    return img_list, E_img_list, gt, absent_label


def run_sequence(seq_name, seed, seq_home, dataset_name, yaml_name, num_gpu=1, checkpoint=None, debug=0, script_name='prompt'):
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
        init_seeds(seed)
    except:
        pass

    seq_txt = seq_name
    # save_name = '{}_ep{}'.format(yaml_name, epoch)
    save_name = '{}'.format(yaml_name)
    save_path = f'./VisEvent_workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'./VisEvent_workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    if script_name == 'spikefet':
        params = parameters.parameters(yaml_name)
        params.debug = debug
        spikefet = SpikeFET(params, checkpoint, dataset_name)  # "VisEvent"
        tracker = SpikeFET_VisEvent(tracker=spikefet)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+ seq_name +'——————————————')
    img_list, E_img_list, gt, absent_label = genConfig(seq_path, dataset_name)
    if absent_label[0] == 0: # first frame is absent in some seqs
        first_present_idx = absent_label.argmax()
        img_list = img_list[first_present_idx:]
        E_img_list = E_img_list[first_present_idx:]
        gt = gt[first_present_idx:]
    if len(img_list) == len(gt):
        result = np.zeros_like(gt)
    else:
        result = np.zeros((len(img_list), 4), dtype=gt.dtype)
    result[0] = np.copy(gt[0])
    toc = 0
    # try:
    for frame_idx, (img_path, E_path) in enumerate(zip(img_list, E_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = _read_image(img_path)
            event_template = _read_image(E_path)
            tracker.initialize(image, event_template, gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            image = _read_image(img_path)
            event_template = _read_image(E_path)
            region, confidence = tracker.track(image, event_template, yaml_name)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class SpikeFET_VisEvent(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, event_template, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, event_template, init_info, idx=0)

    def track(self, img, event_template, yaml_name):
        '''TRACK'''
        outputs = self.tracker.track(img, event_template, yaml_name)
        pred_bbox = outputs['target_bbox']
        return pred_bbox, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on VisEvent dataset.')
    parser.add_argument('--script_name', type=str, default='spikefet', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='spikefet_visevent_tiny', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='VisEvent', help='Name of dataset (VisEvent).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--checkpoint', default='', type=str, help='ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    parser.add_argument('--seed', type=int, default='3407', help='Sequence name for debug.')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))
    # path initialization
    seq_list = None
    if dataset_name == 'VisEvent':
        seq_home = '/home/work/yjj/VisEvent/test'
        with open(join(seq_home, 'list.txt'), 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, args.seed, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.checkpoint, args.debug, args.script_name) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, args.seed, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.checkpoint, args.debug, args.script_name) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {(time.time()-start)/60} minutes!")



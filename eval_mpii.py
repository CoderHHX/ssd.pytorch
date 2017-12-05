"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import MpiiAnnotationTransform, MpiiDetection, BaseTransform, MPII_CLASSES
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--ssd_dim', default=300, type=int, help='SSD300 or SSD512')
parser.add_argument('--vis', default=False, type=str2bool,
                    help='vis the detection results')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)
set_type = 'test'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

def add_rectangles(im, boxes, scores, name_index, min_score = 0.5):

    font_face=cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.6
    color = (0, 0, 255)
    for i, score in enumerate(scores):
        if(score > min_score):
            bbox = np.array(boxes[i, :], dtype = np.int32)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            display_txt = '%s: %.2f'%(MPII_CLASSES[name_index], score)
            cv2.putText(im, display_txt, (bbox[0], bbox[1]), fontFace=font_face, fontScale=font_scale, color=(0,0,255))

    return im




def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05, vis = True):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        image_vis = dataset.pull_image(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            dets = dets.cpu().numpy()
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0]
            cls_dets = np.hstack((boxes, scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets
            if(vis):
                image_vis = add_rectangles(image_vis, boxes, scores, j-1)

        if(vis):
            cv2.imshow("deteation", image_vis)
            cv2.waitKey(0)

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # load net
    num_classes = len(MPII_CLASSES) + 1 # +1 background
    net = build_ssd('test', args.ssd_dim, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = MpiiDetection('/home1/kongtao/workspace/dataset/MPII/images',
                            '/home1/kongtao/workspace/dataset/MPII/mpii_box_keypoints_annotations.json',
                            BaseTransform(args.ssd_dim, dataset_mean),
                            MpiiAnnotationTransform(),
                            is_train=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder,
             net,
             args.cuda, dataset,
             BaseTransform(args.ssd_dim, dataset_mean),
             args.top_k, args.ssd_dim,
             thresh=args.confidence_threshold,
             vis = args.vis)

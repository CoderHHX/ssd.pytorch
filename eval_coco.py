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

from data import COCOAnnotationTransform, COCODetection, BaseTransform, COCO_CLASSES, v2
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
parser.add_argument('--ssd_height', default=512, type=int, help='SSD300 or SSD512')
parser.add_argument('--vis', default=False, type=str2bool,
                    help='vis the detection results')
parser.add_argument('--is_reverse', default=True, type=str2bool, help='Add reverse connections at SSD')

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

def add_rectangles(im, boxes, name_index, color, min_score = 0.5):

    if(boxes is None):
        return im

    font_face=cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.6
    scores = boxes[:, 4]
    for i, score in enumerate(scores):
        if(score > min_score):
            bbox = np.array(boxes[i, :], dtype = np.int32)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=1)
            display_txt = '%s: %.2f'%(COCO_CLASSES[name_index], score)
            cv2.putText(im, display_txt, (bbox[0], bbox[1]), fontFace=font_face, fontScale=font_scale, color=color)
        if(score == -1):
            bbox = np.array(boxes[i, :], dtype = np.int32)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=1)

    return im

def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05, vis = True):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = np.zeros((0, 6), dtype = np.float32)
    all_gts = [[] for _ in range(num_images)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd512_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    gt_file = os.path.join(output_dir, 'gts.pkl')

    # if os.path.exists(det_file) and os.path.exists(gt_file):
    #     with open(det_file, 'rb') as f:
    #         all_boxes = pickle.load(f)
    #     with open(gt_file, 'rb') as f:
    #         all_gts = pickle.load(f)
    #     print('Evaluating detections')
    #     evaluate_detections(all_boxes, all_gts, use_07_metric=False)
    #     return


    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        image_vis = dataset.pull_image(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)
        if(gt is not None):
            gt[:, 0] *= w
            gt[:, 2] *= w
            gt[:, 1] *= h
            gt[:, 3] *= h
            gt[:, 4] = -1
            all_gts[i] = gt


        dets = detections[0, 1, :]
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
        ids = np.zeros_like(scores) + i
        cls_dets = np.hstack((boxes, scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        cls_dets = np.hstack((cls_dets, ids[:, np.newaxis])) \
            .astype(np.float32, copy=False)

        all_boxes = np.vstack((all_boxes, cls_dets))

        if(vis):
            image_vis = add_rectangles(image_vis, gt, 0, color=(255,0,0))
            image_vis = add_rectangles(image_vis, cls_dets, 0, color=(0,0,255))
            cv2.imshow("deteation", image_vis)
            cv2.waitKey(0)

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    with open(gt_file, 'wb') as f:
        pickle.dump(all_gts, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, all_gts, use_07_metric=False)

def evaluate_detections(all_boxes, all_gts, ovthresh = 0.5, use_07_metric = True):

    # first sort the all_boxes
    confidence = all_boxes[:, -2].flatten()
    sorted_ind = np.argsort(-confidence)
    all_boxes = all_boxes[sorted_ind, :]

    nd = len(confidence)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    npos = 0
    for i in range(len(all_gts)):
        npos += np.shape(all_gts[i])[0]


    for d in range(nd):
        bb = all_boxes[d].astype(float)
        ovmax = -np.inf
        BBGT = all_gts[int(all_boxes[d, -1])]
        if np.shape(BBGT)[0] > 0:
            # compute overlaps
            # intersection
            BBGT = BBGT.astype(float)
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) *
                   (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if all_gts[int(all_boxes[d, -1])][jmax, -1] < 0:
                    tp[d] = 1.
                    all_gts[int(all_boxes[d, -1])][jmax, -1] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.


    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    print('AP for {} = {:.4f}'.format('person', ap))

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

if __name__ == '__main__':
    # load net
    num_classes = len(COCO_CLASSES) + 1 # +1 background
    net = build_ssd('test', args.ssd_height, num_classes, reverse=args.is_reverse) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    eval_sets = ['val2017']
    root_path = '/home1/kongtao/workspace/dataset/COCO/2017'

    dataset = COCODetection(root_path, eval_sets,
                            BaseTransform(height=args.ssd_height,
                                          width=args.ssd_height * v2[str(args.ssd_height)]['map_asp'],
                                          mean = dataset_mean),
                            COCOAnnotationTransform(), train=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder,
             net,
             args.cuda, dataset,
             BaseTransform(height=args.ssd_height,
                           width=args.ssd_height * v2[str(args.ssd_height)]['map_asp'],
                           mean = dataset_mean),
             args.top_k, args.ssd_height,
             thresh=args.confidence_threshold,
             vis = args.vis)

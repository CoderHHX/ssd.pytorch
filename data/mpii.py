"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json

MPII_CLASSES = ('person')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class MpiiAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(MPII_CLASSES, range(len(MPII_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for i in range(target.shape[0]):
            bbox = target[i, :]
            bbox[0] = float(bbox[0]) / width
            bbox[1] = float(bbox[1]) / height
            bbox[2] = float(bbox[2]) / width
            bbox[3] = float(bbox[3]) / height
            res += [bbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class MpiiDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, img_folder, json_file, transform=None, target_transform=None,
                 is_train=True, dataset_name='MPII'):
        self.img_folder = img_folder
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.is_train = is_train
        # create train/val split
        with open(json_file) as anno_file:
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, train_flag in enumerate(self.anno):
            if train_flag['train'] == 0:
                self.valid.append(idx)
            else:
                self.train.append(idx)


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

    def pull_item(self, index):
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        gt_boxes = np.zeros((0, 5))
        all_joints = []

        if(len(a['person']) > 1):
            for p_value in a['person']:
                joints = np.array(p_value['joints'])
                all_joints.append(joints)
                keep = np.where(joints[:, 0] > 0)[0]
                joints = joints[keep, :]
                min_all = np.min(joints, axis=0)
                max_all = np.max(joints, axis=0)
                xmin = (min_all[0] -1)
                ymin = (min_all[1] -1)
                xmax = (max_all[0] -1)
                ymax = (max_all[1] -1)
                box = np.array([xmin, ymin, xmax, ymax, 1])
                gt_boxes = np.vstack((gt_boxes, box))
        else:
            joints = np.array(a['person']['joints'])
            all_joints.append(joints)
            keep = np.where(joints[:, 0] > 0)[0]
            joints = joints[keep, :]
            min_all = np.min(joints, axis=0)
            max_all = np.max(joints, axis=0)
            xmin = (min_all[0] -1)
            ymin = (min_all[1] -1)
            xmax = (max_all[0] -1)
            ymax = (max_all[1] -1)
            box = np.array([xmin, ymin, xmax, ymax, 1])
            gt_boxes = np.vstack((gt_boxes, box))

        img_path = os.path.join(self.img_folder, a['image_path'])
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        if self.target_transform is not None:
            gt_boxes = self.target_transform(gt_boxes, width, height)

        if self.transform is not None:
            gt_boxes = np.array(gt_boxes)
            img, boxes, labels = self.transform(img, gt_boxes[:, :4], gt_boxes[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            gt_boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), gt_boxes, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]
        img_path = os.path.join(self.img_folder, a['image_path'])
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

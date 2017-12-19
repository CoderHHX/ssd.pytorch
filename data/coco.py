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
dir_path = os.path.dirname(os.path.realpath(__file__))
import matplotlib.pyplot as plt
from pycocotools.coco import COCO as COCO
from pycocotools import mask as maskUtils
import skimage.io as io
import os.path as osp

COCO_CLASSES = ('person')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class COCOAnnotationTransform(object):
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
            zip(COCO_CLASSES, range(len(COCO_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height, is_train = True):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        key_points_all = np.zeros((0, 3), dtype = np.float32)
        for obj in target:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            bndbox = []
            if is_train:
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    bndbox.append(float(x1)/width)
                    bndbox.append(float(y1)/height)
                    bndbox.append(float(x2)/width)
                    bndbox.append(float(y2)/height)
                    bndbox.append(0)
                    res += [bndbox]
            else:
                if obj['area'] > 32*32:
                    bndbox.append(float(x1)/width)
                    bndbox.append(float(y1)/height)
                    bndbox.append(float(x2)/width)
                    bndbox.append(float(y2)/height)
                    bndbox.append(0)
                    res += [bndbox]

            # keypoints
            key_points = np.array(obj['keypoints']).reshape(-1, 3).astype(np.float32)
            key_points[:, 0] = key_points[:, 0] / width
            key_points[:, 1] = key_points[:, 1] / height
            key_points_all = np.vstack((key_points_all, key_points))



        if(len(res) == 0):
            return res, key_points_all, False

        return res, key_points_all, True


class COCODetection(data.Dataset):
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

    def __init__(self, root, image_sets, transform=None, target_transform=None,
                 dataset_name='COCO14', train = True):

        self.root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.imgIds = []
        self.image_paths = []
        self.cocos = []
        self.data_index = []
        self.is_train = train
        for i, image_set in enumerate(self.image_sets):
            json_file = osp.join(self.root,'annotations', 'person_keypoints_'+image_set+'.json')
            coco = COCO(json_file)
            self.cocos.append(coco)
            imgIds = self.get_anns(coco)
            for img_id in imgIds:
                self.data_index.append(i)
                self.imgIds.append(img_id)
                self.image_paths.append(self.image_path_from_index(image_set, img_id))


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.imgIds)

    def image_path_from_index(self, dataset, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = osp.join(self.root, dataset, file_name)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def get_anns(self, coco):
        imgIds = coco.getImgIds()
        valid_imgIds = []
        for i, imgId in enumerate(imgIds):
            if(self.is_train == False):
                valid_imgIds.append(imgId)
                continue

            img = coco.loadImgs(imgId)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
            anns = coco.loadAnns(annIds)
            # Consider only images with people
            has_people = len(anns) > 0
            if not has_people:
                continue

            width = img["width"]
            height = img["height"]
            obj_num = 0
            for obj in anns:
                x1 = np.max((0, obj['bbox'][0]))
                y1 = np.max((0, obj['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj_num = obj_num + 1

            if(obj_num < 1):
                continue

            valid_imgIds.append(imgId)
        return  valid_imgIds

    def pull_item(self, index):
        coco = self.cocos[self.data_index[index]]
        imgId =  self.imgIds[index]
        img_infos = coco.loadImgs(imgId)[0]

        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        annIds = coco.getAnnIds(imgIds=img_infos['id'], iscrowd=False)
        target = coco.loadAnns(annIds)
        key_points = []
        if self.target_transform is not None:
            target, key_points, has_boxes = self.target_transform(target, width, height, self.is_train)

        if self.transform is not None:
            target = np.array(target)
            if(self.is_train):
                img, target, labels, key_points = self.transform(img, target[:, :4], target[:, 4], key_points)
            else:
                if(np.shape(target)[0] == 0):
                    img, target, labels = self.transform(img, None, None)
                else:
                    img, target, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            if(target is not None):
                target = np.hstack((target, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
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
        img_path = self.image_paths[index]
        return cv2.imread(img_path)



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def image_path_from_index(root, dataset, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = (str(index).zfill(12) + '.jpg')
    image_path = osp.join(root, dataset, file_name)
    assert osp.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
    return image_path

if __name__ == '__main__':

    json_file = '/home1/kongtao/workspace/dataset/COCO/2017/annotations/person_keypoints_train2017.json'
    coco = COCO(json_file)
    imgIds = coco.getImgIds()
    num_joints = 17

    person_n = 0
    valid_imgIds = []
    for i, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgId)[0]
        im_size = [3, img["height"], img["width"]]
        im_path = img["file_name"]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
        anns = coco.loadAnns(annIds)
        # Consider only images with people
        has_people = len(anns) > 0
        if not has_people:
            continue

        # for an in anns:
        #     x = an['bbox'][0]
        #     y = an['bbox'][1]
        #     w = an['bbox'][2]
        #     h = an['bbox'][3]
        #     num_keypoints = an['num_keypoints']
        valid_imgIds.append(imgId)
        image_path = image_path_from_index('/home1/kongtao/workspace/dataset/COCO/2017', 'train2017', imgId)
        I = io.imread(image_path)
        plt.imshow(I)
        coco.showAnns(anns)
        plt.show()
        if(i % 100 ==0):
            print(i)








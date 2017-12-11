import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, MpiiAnnotationTransform, MpiiDetection, detection_collate, MPII_CLASSES
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=9, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--ssd_height', default=512, type=int, help='SSD300 or 512')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = v2

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# train_sets = 'train'
ssd_dim = args.ssd_height
means = (104, 117, 123)  # only support voc now
num_classes = len(MPII_CLASSES) + 1
batch_size = args.batch_size
max_iter = args.iterations
weight_decay = args.weight_decay
stepvalues = (60, 90)
gamma = args.gamma
momentum = args.momentum

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', ssd_dim, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cfg[str(ssd_dim)]['variance'], args.cuda)

dataset = MpiiDetection('/home1/kongtao/workspace/dataset/MPII/images',
                        '/home1/kongtao/workspace/dataset/MPII/mpii_box_keypoints_annotations.json',
                        SSDAugmentation(height=args.ssd_height, width=args.ssd_height * v2[str(args.ssd_height)]['map_asp'], mean = means),
                        MpiiAnnotationTransform())
data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

def train_one_iters(iteration, lr, clip = 4.0):

    loc_loss = 0
    conf_loss = 0
    t0 = time.time()
    for i, (images, targets) in enumerate(data_loader):
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]

        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        torch.nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if 0:
            print('batch: '+ repr(i) + '/' +repr(len(data_loader)) +' || Loss: %.4f' % (loss.data[0]), end='\n')
    t1 = time.time()
    loc_loss = loc_loss / len(data_loader)
    conf_loss = conf_loss / len(data_loader)
    print('iter ' + repr(iteration) + ' || LR: {:.5f} || Timer: {:.2f} sec.'.format(lr, (t1 - t0)), end=' || ')
    print('loc_loss: {:.4f} || conf_loss: {:.4}.'.format(loc_loss, conf_loss))

    if((iteration+1) % 10 == 0):
        torch.save(ssd_net.state_dict(), 'weights/ssd'+str(ssd_dim)+ '_MPII_' +
                           repr(iteration+1) + '.pth')

def train_net():
    lr = args.lr
    net.train()
    for epoch in range(args.start_iter, max_iter):
        lr = adjust_learning_rate(optimizer, epoch, lr, stepvalues, args.gamma)
        train_one_iters(epoch, lr)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr
if __name__ == '__main__':
    train_net()

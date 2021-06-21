from abc import abstractmethod
import argparse
from genericpath import exists
from logging import Logger
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn.init as init

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from logger import logger
from ptflops import get_model_complexity_info

model_names = ["FibNet"]
dataset_choices = ["imagenet", "cifar100"]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='FibNet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: FibNet)')
parser.add_argument('--ds', '--dataset_name', type = str, dest = 'dataset_name', default = 'Imagenet',
                    help = 'Name of dataset to be used.',
                    choices = dataset_choices)
parser.add_argument('--nb', '--n-blocks', default=0, type=int, 
                    help='number of fibNet blocks', dest='n_blocks')
parser.add_argument('--bd', '--block-depth', default=0, type=int, 
                    help='FibNet Block Depth', dest='block_depth' )
parser.add_argument('--r1', default=0.618, type=float, 
                    help='First ratio for FibNet')
parser.add_argument('--r2', default=3.414, type=float, 
                    help='First ratio for FibNet')
parser.add_argument('--use_conv_cat', default=True, type=bool, dest= 'use_conv_cat',
                    help= 'For FibNet to choose wether using conv_cat (True) or maxpooling2d (False)')
parser.add_argument('--cl', '--num_class', default=100, type=int, 
                    help='Number of Class', dest='num_class' )
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_mode', default="step", type=str,
                    help='Learning rate decay mode', choices=['cosine','step'],  dest='lr_mode')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--log_path', default = None, type = str, dest = "log_path",
                        help='Sets the tensorboard logging.')


best_acc1 = 0
train_steps = 0
val_steps = 0 
def main():
    args = parser.parse_args()
    
    if args.log_path is not None:
        if(args.arch == "FibNet"):
            log = logger.Create(os.path.join(args.log_path,args.arch+str(args.n_blocks)+'x'+str(args.block_depth)))
        else:
            log = logger.Create(os.path.join(args.log_path,args.arch))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')

    main_worker(args.gpu, log, args)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain = nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m,nn.BatchNorm2d):
        torch.nn.init.ones_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain = nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(m.bias)     

def main_worker(gpu, log, args):
    global best_acc1
    global train_steps
    global val_steps
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    
    #TODO: work on this code block to enable using pretrained weights.
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     if args.arch == "FibNet":
    #         from models.FibNet import FibNet
    #         model = FibNet(in_channels = 3, out_channels = args.num_class, num_blocks = args.n_blocks, block_depth = args.block_depth, pretrained=True)
    #     elif args.arch == "MobileNetv2":
    #         from models.MobileNetv2 import MobileNetv2
    #         model = MobileNetv2(args.num_class)            
        # else:

    print("=> creating model '{}'".format(args.arch))
    if args.arch == "FibNet":
        from models.FibNet import FibNet
        model = FibNet(in_channels = 3, out_channels = args.num_class, num_blocks = args.n_blocks, block_depth = args.block_depth, pretrained=False, use_conv_cat=args.use_conv_cat, r1=args.r1, r2=args.r2)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            train_steps = checkpoint['train_steps']
            val_steps = checkpoint['val_steps']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset_name == "imagenet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.dataset_name == "cifar100":
        transform = transforms.Compose(
                        [transforms.ToTensor(),transforms.Resize(64),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

        trainset = datasets.CIFAR100(root = args.data, train = True, transform = transform ,download = True)
        testset = datasets.CIFAR100(root =  args.data, train = False, transform = transform ,download = True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers,
                                                pin_memory=True)
        val_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=False,  num_workers=args.workers, 
                                                pin_memory=True)  

    if args.evaluate:
        validate(val_loader, model, criterion, val_steps, log, args)
        return   
    start_time = datetime.now()
    args.start_time = start_time
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_steps =train(train_loader, model, criterion, optimizer, epoch, train_steps, log, args)

        # evaluate on validation set
        acc1, acc2, val_steps = validate(val_loader, model, criterion, val_steps, log, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        #for file path
        if(args.arch == "FibNet"):
            model_name = args.arch+str(args.n_blocks)+'x'+str(args.block_depth)
        else:
            model_name = args.arch                                    
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'train_steps': train_steps,
                'val_steps': val_steps
            }, is_best, model_name)

    if args.dataset_name == 'cifar100':
        image_shape = (3,64,64)
    elif args.dataset_name == 'imagenet':
        image_shape = (3,224,224)

    macs, params= get_model_complexity_info(model, image_shape, as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    log.h_params(args.__dict__,{'Top_1':acc1, 'Top_5':acc2, 'GMacs': float(macs[:-4]), 'Params': float(params[:-2])},"training_config")

def train(train_loader, model, criterion, optimizer, epoch, train_steps, log, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.4f')
    top5 = AverageMeter('Acc@5', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        train_steps+=1
        if(args.log_path is not None):
            #raw
            log.train_top1(top1.val, train_steps)
            log.train_top5(top5.val, train_steps)
            log.train_loss(losses.val, train_steps)
            #averages
            log.train_top1_avg(top1.avg, train_steps)
            log.train_top5_avg(top5.avg, train_steps)
            log.train_mloss(losses.avg, train_steps)
            
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i,args)
    return train_steps


def validate(val_loader, model, criterion, val_steps, log, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            val_steps+=1
            if(args.log_path is not None):
                #raw
                log.val_top1(top1.val,val_steps)
                log.val_top5(top5.val,val_steps)
                log.val_loss(losses.val, val_steps)

                #averages
                log.val_top1_avg(top1.avg,val_steps)
                log.val_top5_avg(top5.avg,val_steps)
                log.val_mloss(losses.avg, val_steps)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i,args)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, val_steps


def save_checkpoint(state, is_best, model_name, filename='checkpoint.pth.tar'):
    checkpoint_fp = os.path.join("checkpoints", model_name)
    weights_fp = os.path.join("weights", model_name)

    if(os.path.exists(checkpoint_fp) == False):
        os.mkdir(checkpoint_fp)
    if(os.path.exists(weights_fp) == False):
        os.mkdir(weights_fp)

    checkpoint_fp =  os.path.join(checkpoint_fp,filename)
    weights_fp  = os.path.join(weights_fp,filename)
    torch.save(state, checkpoint_fp)
    if is_best:
        shutil.copyfile(checkpoint_fp, weights_fp)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        elapsed_time = datetime.now() - args.start_time
        print('\t'.join(entries) + '\tElapsed Time: ' +str(elapsed_time))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    if(args.lr_mode == 'cosine'):
        #Cosine learning rate decay
        lr = 0.5 * args.lr  * (1 + np.cos(np.pi * (epoch)/ args.epochs ))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import torch.nn as nn
import torch.nn.init as init
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

import time
import os
import copy
import argparse
import shutil
import sys
import math


global device
top5_flag = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
log = True
timestr = time.strftime("%m%d%H%M", time.localtime(time.time()))
if log:
    fp = open('log/param'+timestr, 'a')
    facc = open('log/acc'+timestr, 'a')
    ftrain = open('log/train'+timestr, 'a')
## 
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

class chvgg(nn.Module):
    def __init__(self, features, num_classes=10):
        super(chvgg, self).__init__()   
        l = list(features.children())
        self.features = torch.nn.Sequential(*l, nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)))
        
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
#class cifarresnet(nn.module):
    

class chresnet(nn.Module):
    def __init__(self, model, num_classes=10):
        super(chresnet,self).__init__() 
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)        
        #self.model = model
        self.fconv = nn.Conv2d(2048, num_classes, kernel_size=1,stride=1, bias=False)        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
    
        x = self.fconv(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
           
        return x    
##
def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

jfiods = os.popen('stty size', 'r').read()

#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
term_width = 300

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
##
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

##
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

##
def replace_layers(model, idx, ridx, rlayers):
    if idx in ridx:
        return rlayers[ridx.index(idx)]
    return model[idx]

##
def replace_Sequential(features, idx, rlayers):
    return torch.nn.Sequential(
            *(replace_layers(features, i, idx,  rlayers) \
              for i, _ in enumerate(features)))

##
def copy_conv(conv_old, channel,  out = True, groups = 1):
    if groups == 1:
        if out:
            conv_new = \
                nn.Conv2d(in_channels = conv_old.in_channels, 
                          out_channels = conv_old.out_channels-channel.size()[0], 
                          kernel_size = conv_old.kernel_size, stride = conv_old.stride,
                          padding = conv_old.padding,  dilation = conv_old.dilation, 
                          groups = conv_old.groups , bias=True if conv_old.bias is not None else False
                                )        
            l = torch.arange(conv_old.out_channels, dtype=torch.long).to(device)
            l_ = torch.ones(conv_old.out_channels, dtype=torch.uint8).to(device)
            l_[channel] = 0
            l = torch.masked_select(l, l_)
            
            conv_new.weight.data = torch.index_select(conv_old.weight.data, 0, l)
            if conv_new.bias is not None: 
                conv_new.bias.data = torch.index_select(conv_old.bias.data, 0, l)
            conv_new = conv_new.to(device)
        else:
            conv_new = \
                nn.Conv2d(in_channels = conv_old.in_channels-channel.size()[0], 
                          out_channels = conv_old.out_channels, 
                          kernel_size = conv_old.kernel_size, stride = conv_old.stride,
                          padding = conv_old.padding,  dilation = conv_old.dilation, 
                          groups = conv_old.groups ,  
                                )         
            l = torch.arange(conv_old.in_channels, dtype=torch.long).to(device)
            l_ = torch.ones(conv_old.in_channels, dtype=torch.uint8).to(device)
            l_[channel] = 0
            l = torch.masked_select(l, l_)            
            conv_new.weight.data = torch.index_select(conv_old.weight.data, 1, l)
            #conv_new.bias.data = torch.index_select(conv_old.bias.data, 0, l)   
            conv_new = conv_new.to(device)
    else:
        ## unstable
        conv_new = \
            nn.Conv2d(in_channels = conv_old.in_channels-len(channel), 
                      out_channels = conv_old.out_channels-len(channel), 
                      kernel_size = conv_old.kernel_size, stride = conv_old.stride,
                      padding = conv_old.padding,  dilation = conv_old.dilation, 
                      groups = conv_old.groups-len(channel) ,  #bias=conv_old.bias
                            )           
        conv_new = conv_new.to(device)
    
    return conv_new

##
def copy_bn(bn_old, channel):
    bn_new = \
        nn.BatchNorm2d(num_features = bn_old.num_features-channel.size()[0], 
                       eps = bn_old.eps, momentum = bn_old.momentum)
    l = torch.arange(bn_old.num_features, dtype=torch.long).to(device)
    l_ = torch.ones(bn_old.num_features, dtype=torch.uint8).to(device)
    l_[channel] = 0
    l = torch.masked_select(l, l_)    
    
    bn_new.weight.data = torch.index_select(bn_old.weight.data, 0, l)    
    bn_new.bias.data = torch.index_select(bn_old.bias.data, 0, l)    
    bn_new = bn_new.to(device)
    return bn_new

    

##  train  
def get_cifar(root='../data'):
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    trainset = torchvision.datasets.CIFAR10(root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=12)
    
    testset = torchvision.datasets.CIFAR10(root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=12)
    dataloaders = {'train':trainloader, 'val': testloader}
    return dataloaders 


def get_data(root='../data/VOC2012'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(root, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=26,
                                                 shuffle=True, num_workers=12)
                  for x in ['train', 'val']}    
    return dataloaders

def info(mode, top5_flag, epoch=0, i=0, iters=0, batch_time=0, data_time=0, losses=0, top1=0, top5=0):
    if mode == "train":
        if top5_flag:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               epoch, i, iters, batch_time=batch_time,
               data_time=data_time, loss=losses, top1=top1, top5=top5))    
        else:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' .format(
               epoch, i, iters, batch_time=batch_time,
               data_time=data_time, loss=losses, top1=top1))                
    else:
        if top5_flag:
            print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               i, iters, batch_time=batch_time, loss=losses,
               top1=top1, top5=top5))
        else:
            print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
               i, iters, batch_time=batch_time, loss=losses,
               top1=top1))                    

def train(args, train_loader, model, criterion, optimizer, epoch, small_show = False):
    ## initial the param
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    if top5_flag:
        top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # = target.cuda(async=True)
        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        if top5_flag:   prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        else:  prec1 = accuracy(output.data, target, topk=(1))
        
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        if top5_flag: top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not small_show:
                info('train', top5_flag, epoch, i, len(train_loader), batch_time=batch_time, \
                 data_time=data_time, losses=losses, top1=top1,top5= top5)    


def validate(args, val_loader, model, criterion, small_show = False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    if top5_flag: top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        if top5_flag:  prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        else:  prec1 = accuracy(output.data, target, topk=(1))
        
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        if top5_flag: top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not small_show:
                info('Test', top5_flag, i, len(val_loader), batch_time=batch_time, \
                     losses=losses, top1=top1, top5=top5)

    if top5_flag:
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5), end=' --')
    else:
        print(' * Prec@1 {top1.avg:.3f}' .format(top1=top1))        

    return top1.avg


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model')
        
def save(state, is_best, path, arch):
    name = path+arch+'.pth'
    torch.save(state, name)
    if is_best:
        torch(state, path+arch+'best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(args, optimizer, epoch, pitch=20):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // pitch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

def train_epochs(args, dataloaders, model, criterion, optimizer, epochs, pitch=30,start_epoch=0, store = False, best_prec = 0, small_show = False):
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(args, optimizer, epoch, pitch=pitch)
        
        tic = time.time()
        # train for one epoch
        train(args, dataloaders['train'], model, criterion, optimizer, epoch, small_show = small_show)        
        # evaluate on validation set
        prec1 = validate(args, dataloaders['val'], model, criterion, small_show = small_show)
        print('time:', time.time()-tic)
        if store:
            #save(state, is_best, path, arch)
            print('best_prec', best_prec)
            is_best = prec1 > best_prec #best_prec = max(prec1, best_prec)
            save_checkpoint(model, is_best,args.path)        
            best_prec = max(prec1, best_prec)
            #return best_prec
            #{ 'epoch': epoch + 1, 'arch': args.arch,\
                                #'state_dict': model.state_dict(), 'best_prec1': best_prec,\
                                #'optimizer' : optimizer.state_dict(), }        
        #is_best = prec1 > best_prec #best_prec = max(prec1, best_prec)
        #save_checkpoint({ #'epoch': epoch + 1, #'arch': args.arch,\
        #'state_dict': model.state_dict(), #'best_prec1': best_prec,\
        #'optimizer' : optimizer.state_dict(), #}, is_best,args.path)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res    


def get_mean_std(model, trainset):
    pass

#def cal_layer(x, layer = None, idx = 0):
    #if idx == 0:
        #print(layer)
        #return layer(x)
    #else:
        #print(layer[idx-1])
        #return layer[:idx](x)
    
def cal_feature(x, num, l, cal_mean=0, cal_std=0, mean = 0, std = 0):
    x_view = x.view(x.size(0), x.size(1),  -1)
    x_mean = torch.mean(x_view, 2)
    x_std = torch.std(x_view, 2)
    #print('mean', x_mean.mean(), ' std', x_std.mean())
    cal_mean = (num * cal_mean + l * x_mean.mean()) / (num + l)
    cal_std = (num * cal_std + l * x_std.mean()) / (num + l)
    
    x_mean = (num * mean + l * x_mean.mean(dim=0)) / (num + l)
    x_std = (num * std + l * x_std.mean(dim=0)) / (num + l)
    #num = num + l
    return x_mean, x_std, cal_mean, cal_std, num+l

def cal_goal(cal_mean, cal_std):
    
    return goal
    

def filter_cal_threshold(model, trainset, layer_cur = 0,  classes = 120, epochs = 0, diff = 0, parm={'alpha':1, 'beta':1, 'gamma':1, 'scale':1}):
    mean = [0.14, 1, 1, 1,
            0.15, 1, 1, 1,
            0.16, 1, 1, 1,
            0.13, 1, 1, 1,
            0.18, 1, 1, 1,
            0.29, 1, 1, 1,
            0.27, 1, 1, 1,
            0, 1, 1, 1,]
    std = [0.18, 1, 1, 1,
            0.2, 1, 1, 1,
            0.24, 1, 1, 1,
            0.28, 1, 1, 1,
            0.29, 1, 1, 1,
            0.33, 1, 1, 1,
            0.22, 1, 1, 1,
            0, 1, 1, 1,]    
    #std = [0.08, 1, 1, 1,
           #0.03, 1, 1, 1,
            #0.04, 1, 1, 1,
            #0.08, 1, 1, 1,
            #0.07, 1, 1, 1,
            #0.03, 1, 1, 1,
            #0.02, 1, 1, 1,
            #0, 1, 1, 1,]     
    print('fileter pruning begin--------------------')
    layer_find = None
    num = torch.tensor(0, dtype = torch.float).to(device)
    cal_mean = torch.zeros_like(num)
    cal_std = torch.zeros_like(num)
   
    tic = time.time()
    layer_len = len(model.features._modules.items())
    for i, (xx, y) in enumerate(trainset):
        xx, y = xx.to(device), y.to(device)
        for idx, layer in enumerate(model.features.children()):
            toc = time.time()
            
            # search for current convolution 
            if isinstance(layer, nn.Conv2d):
                if not layer_find:  conv = idx
                else:  
                    if idx < layer_cur: conv = idx
                    
            if isinstance(layer, nn.ReLU):
                ## initial cur_layer   the activation layer 
                if not layer_find:
                    if layer_cur < idx :    
                        layer_cur = idx
                        layer_find = True
                if idx == layer_cur:   # fine the current activation layer
                    x = model.features[:idx+1](xx).detach()
                    x_view = x.view(x.size(0), x.size(1),  -1)
                    x_mean = torch.mean(x_view, 2)
                    x_std = torch.std(x_view, 2)    
                    break
            #print(time.time()-toc,'--', idx, ' : ', layer)
            
        ## darknet features -> output
        x = model.features(xx).detach()
        torch.cuda.empty_cache()
        pre_y = x.view(-1, classes)
        pred = pre_y.argmax(dim=1).eq(y)
        pre_y = nn.Softmax(dim=1)(pre_y)
        one_hot = torch.zeros_like(pre_y).scatter_(1, y.view(-1, 1).long(), -1)
        pre_y.add_(one_hot)
        y_ = pre_y.norm(dim=1)
        
        ### deal with mean and std
        cal_mean = (num * cal_mean + y.size(0) * x_mean.mean(dim=0)) / (num + y.size(0))
        cal_std = (num * cal_std + y.size(0) * x_std.mean(dim=0)) / (num + y.size(0))
        num += y.size(0)
        
    
    if epochs:
    # 鏇存柊瓒呭弬鏁扮殑锟�        if epochs >= mean[conv]:
        parm['alpha'] = mean[conv]
    else:
        parm['alpha'] = epochs
        
    if epochs + diff >= std[conv]:
        parm['beta'] = std[conv]
    else:
        parm['beta'] = epochs + diff
    
    ## mean
    #print ('mean method - alpha', parm['alpha'])
    #goal = 1- parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    ## std
    print ('std method - beta', parm['beta'])
    goal = cal_std / parm['beta'] - 1
    ## std + mean
    #print('mean - ', parm['alpha'], ' and std - ', parm['beta'])
    #goal = cal_std / parm['beta'] - parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    
    print('total', time.time()-tic)
    return conv, layer_cur, goal, parm


def filter_cal(model, trainset, layer_cur = 0,  classes = 120, parm={'alpha':1, 'beta':1, 'gamma':1, 'scale':1}):
    print('fileter pruning begin--------------------')
    layer_find = None
    #model.eval()
    num = torch.tensor(0, dtype = torch.float).to(device)
    cal_mean = torch.zeros_like(num)
    cal_std = torch.zeros_like(num)
    #x_mean = torch.zeros_like(num)
    #x_std = torch.zeros_like(num)
    #conv=torch.zeros_like(num)
   
    tic = time.time()
    layer_len = len(model.features._modules.items())
    for i, (xx, y) in enumerate(trainset):
        xx, y = xx.to(device), y.to(device)
        for idx, layer in enumerate(model.features.children()):
            toc = time.time()
            
            # search for current convolution 
            if isinstance(layer, nn.Conv2d):
                if not layer_find:  conv = idx
                else:  
                    if idx < layer_cur: conv = idx
                    
            if isinstance(layer, nn.ReLU):
                ## initial cur_layer   the activation layer 
                if not layer_find:
                    if layer_cur < idx :    
                        layer_cur = idx
                        layer_find = True
                if idx == layer_cur:   # fine the current activation layer
                    x = model.features[:idx+1](xx).detach()
                    x_view = x.view(x.size(0), x.size(1),  -1)
                    x_mean = torch.mean(x_view, 2)
                    #print ("x_mean-", x_mean)
                    x_std = torch.std(x_view, 2)    
                    break
            #print(time.time()-toc,'--', idx, ' : ', layer)
            
        ## darknet features -> output
        x = model.features(xx).detach()
        torch.cuda.empty_cache()
        pre_y = x.view(-1, classes)
        pred = pre_y.argmax(dim=1).eq(y)
        pre_y = nn.Softmax(dim=1)(pre_y)
        one_hot = torch.zeros_like(pre_y).scatter_(1, y.view(-1, 1).long(), -1)
        pre_y.add_(one_hot)
        y_ = pre_y.norm(dim=1)
        
        #print ("x_mean+", x_mean)
        ### deal with mean and std
        cal_mean = (num * cal_mean + y.size(0) * x_mean.mean(dim=0)) / (num + y.size(0))
        cal_std = (num * cal_std + y.size(0) * x_std.mean(dim=0)) / (num + y.size(0))
        num += y.size(0)
        
        #print('cal_mean:', cal_mean)
        #print('cal_std: ', cal_std)
        #print('mean:1-', x_mean.mean(dim=0)[0], ' 2-', cal_mean[0], 
        #     '   std:1-', x_std.mean(dim=0)[0], ' 2-', cal_std[0])

    ## update parm
    #parm['alpha'] = parm['scale'] * cal_mean.mean()
    #parm['beta'] = parm['scale'] * cal_std.mean()    
    
    ## mean
    #print ('mean method - alpha', parm['alpha'])
    #goal = 1- parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    ## std
    #print ('std method - beta', parm['beta'])
    #goal = cal_std / parm['beta'] - 1
    ## std + mean
    print('mean - ', parm['alpha'], ' and std - ', parm['beta'])
    goal = cal_std / parm['beta'] - parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    
    ### update parm
    #parm['alpha'] = parm['scale'] * cal_mean.mean()
    #parm['beta'] = parm['scale'] * cal_std.mean()
        
    ##-------goal = x_mean.sum(0)
    #print('alpha:', parm['alpha'], '--cal_mean:', cal_mean)
    #print('beta:', parm['beta'], 'cal_std: ', cal_std) 
    print('total', time.time()-tic)
    return conv, layer_cur, goal, parm
        
def filter_vgg_cal(model, trainset, layer_cur = 0,  classes = 120, parm={'alpha':1, 'beta':1, 'gamma':1, 'scale':1}):
    print('fileter pruning begin--------------------')
    layer_find = None
    #model.eval()
    num = torch.tensor(0, dtype = torch.float).to(device)
    cal_mean = torch.zeros_like(num)
    cal_std = torch.zeros_like(num)
   
    tic = time.time()
    layer_len = len(model.features._modules.items())
    for i, (xx, y) in enumerate(trainset):
        xx, y = xx.to(device), y.to(device)
        for idx, layer in enumerate(model.features.children()):
            toc = time.time()
            
            # search for current convolution 
            if isinstance(layer, nn.Conv2d):
                if not layer_find:  conv = idx
                else:  
                    if idx < layer_cur: conv = idx
                    
            if isinstance(layer, nn.ReLU):
                ## initial cur_layer   the activation layer 
                if not layer_find:
                    if layer_cur < idx :    
                        layer_cur = idx
                        layer_find = True
                if idx == layer_cur:   # fine the current activation layer
                    x = model.features[:idx+1](xx).detach()
                    x_view = x.view(x.size(0), x.size(1),  -1)
                    x_mean = torch.mean(x_view, 2)
                    #print ("x_mean-", x_mean)
                    x_std = torch.std(x_view, 2)    
                    break
            #print(time.time()-toc,'--', idx, ' : ', layer)
            
        ## darknet features -> output
        #x = model.features(xx).detach()
        x = model(xx).detach()
        torch.cuda.empty_cache()
        pre_y = x.view(-1, classes)
        pred = pre_y.argmax(dim=1).eq(y)
        pre_y = nn.Softmax(dim=1)(pre_y)
        one_hot = torch.zeros_like(pre_y).scatter_(1, y.view(-1, 1).long(), -1)
        pre_y.add_(one_hot)
        y_ = pre_y.norm(dim=1)
        
        #print ("x_mean+", x_mean)
        ### deal with mean and std
        cal_mean = (num * cal_mean + y.size(0) * x_mean.mean(dim=0)) / (num + y.size(0))
        cal_std = (num * cal_std + y.size(0) * x_std.mean(dim=0)) / (num + y.size(0))
        num += y.size(0)
    
    ## mean
    goal = 1- parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    ## std
    #goal = cal_std / parm['beta'] - 1
    ## std + mean
    #goal = cal_std / parm['beta'] - parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    
        
    ##-------goal = x_mean.sum(0)
    print('alpha:', parm['alpha'], '--cal_mean:', cal_mean)
    print('beta:', parm['beta'], 'cal_std: ', cal_std) 
    print('total', time.time()-tic)
    return conv, layer_cur, goal, parm 


def resnet_cal(model, layer_cur):# layer_cur = 0,  classes = 120, parm={'alpha':1, 'beta':1, 'gamma':1, 'scale':1}):
    print('fileter pruning begin--------------------')
    #model.eval()
    num = torch.tensor(0, dtype = torch.float).to(device)
    cal_mean = torch.zeros_like(num)
    cal_std = torch.zeros_like(num)
    reslayer = [model.layer1, model.layer2, model.layer3, model.layer4]
    #layer = None
    #item = list(model.children())
    #layer_len = len(item)
    #conv_cur = item[layer_idx[0]]
    
    #layer_len = len(model.features._modules.items())
    xx = torch.randn([2,3,224,224])
    
    if layer_cur[0] == 0:
        print(model.conv1)
        print(model.bn1)
        print(model.relu)
        y = model.relu(model.bn1(model.conv1(xx)))
        #continue
    else:
        idx = layer_cur[0]-1
        print(model.conv1)
        print(model.bn1)
        print(model.relu)
        print(model.maxpool)
        xx = model.maxpool(model.relu(model.bn1(model.conv1(xx))))
        for l in range(idx):
            print(reslayer[l])
            xx = reslayer[l](xx)
            
        xx = reslayer[idx][:layer_cur[1]](xx)
        print(reslayer[idx][:layer_cur[1]])
        blayer = reslayer[idx][layer_cur[1]]
        if layer_cur[2] == 1:
            print(blayer.conv1)
            print(blayer.bn1)
            xx = nn.ReLU(blayer.bn1(blayer.conv1(xx)))
        elif layer_cur[2] == 2:
            if blayer.conv3:
                print(blayer.conv1)
                print(blayer.bn1)
                print(blayer.conv2)
                print(blayer.bn2)                
                xx = nn.ReLU(blayer.bn2(blayer.conv2(nn.ReLU(blayer.bn1(blayer.conv1(xx))))))
            else:
                print(blayer)
                xx = blayer(xx)
        else:
            print(blayer)
            xx = blayer(xx)
        
        
        
        
    ##for idx 
    
    #for idx, layer in enumerate(model.children()):
        #print(idx, layer)
        #if isinstance(layer, nn.conv2d):
            #print(idx, layer)
            #y = model.relu(model.bn1(model.conv1(xx)))
        #if isinstance(layer, nn.Sequential):
            #for idx2, sublayer in enumerate(layer.children()):
                #if isinstance(sublayer, nn.Sequential):
        # search for current convolution 
        #if isinstance(layer, nn.Conv2d):
            #if not layer_find:  conv = idx
            #else:  
                #if idx < layer_cur: conv = idx
                
        #if isinstance(layer, nn.ReLU):
            ### initial cur_layer   the activation layer 
            #if not layer_find:
                #if layer_cur < idx :    
                    #layer_cur = idx
                    #layer_find = True
            #if idx == layer_cur:   # fine the current activation layer
                #x = model.features[:idx+1](xx).detach()
                #x_view = x.view(x.size(0), x.size(1),  -1)
                #x_mean = torch.mean(x_view, 2)
                ##print ("x_mean-", x_mean)
                #x_std = torch.std(x_view, 2)    
                #break
        #print(time.time()-toc,'--', idx, ' : ', layer)
        
    ## darknet features -> output
    #x = model.features(xx).detach()
    #torch.cuda.empty_cache()
    #pre_y = x.view(-1, classes)
    #pred = pre_y.argmax(dim=1).eq(y)
    #pre_y = nn.Softmax(dim=1)(pre_y)
    #one_hot = torch.zeros_like(pre_y).scatter_(1, y.view(-1, 1).long(), -1)
    #pre_y.add_(one_hot)
    #y_ = pre_y.norm(dim=1)
    
    #print ("x_mean+", x_mean)
    #### deal with mean and std
    #cal_mean = (num * cal_mean + y.size(0) * x_mean.mean(dim=0)) / (num + y.size(0))
    #cal_std = (num * cal_std + y.size(0) * x_std.mean(dim=0)) / (num + y.size(0))
    #num += y.size(0)
    
    #print('cal_mean:', cal_mean)
    #print('cal_std: ', cal_std)
    #print('mean:1-', x_mean.mean(dim=0)[0], ' 2-', cal_mean[0], 
    #     '   std:1-', x_std.mean(dim=0)[0], ' 2-', cal_std[0])

    ## update parm
    #parm['alpha'] = parm['scale'] * cal_mean.mean()
    #parm['beta'] = parm['scale'] * cal_std.mean()    
    
    ## mean
    #print ('mean method - alpha', parm['alpha'])
    #goal = 1- parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    ## std
    #print ('std method - beta', parm['beta'])
    #goal = cal_std / parm['beta'] - 1
    ## std + mean
    #print('mean - ', parm['alpha'], ' and std - ', parm['beta'])
    #goal = cal_std / parm['beta'] - parm['alpha'] * (cal_mean + 0.0001).reciprocal()
    
    ### update parm
    #parm['alpha'] = parm['scale'] * cal_mean.mean()
    #parm['beta'] = parm['scale'] * cal_std.mean()
        
    ##-------goal = x_mean.sum(0)
    #print('alpha:', parm['alpha'], '--cal_mean:', cal_mean)
    #print('beta:', parm['beta'], 'cal_std: ', cal_std) 
    #print('total', time.time()-tic)
    #return conv, layer_cur, goal, parm

def get_nextlayer(model, idx, type='resnet'):
    if type == 'resnet':
        reslayer = [model.layer1, model.layer2, model.layer3, model.layer4]
        if idx[0] == 0:
            idx = [1, 0, 1]
        else:
            if idx[2] == 1:
                idx[2] = 2
            elif idx[2] == 2:
                if reslayer[idx[0]-1][idx[1]].conv3 is not None:
                    idx[2] = 0
                else:
                    blen = len(reslayer[idx[0]-1])
                    if idx[1] < blen-1:
                        idx[1]+= 1
                        idx[2] = 1
                    elif idx[1] == blen-1:
                        if idx[0] == 4:
                            idx = [0, 0, 0]
                        else:
                            idx = [idx[0]+1, 0, 1]
            else:
                blen = len(reslayer[idx[0]-1])
                if idx[1] < blen-1:
                    idx[1]+= 1
                    idx[2] = 1
                elif idx[1] == blen-1:
                    if idx[0] == 4:
                        idx = [0, 0, 0]
                    else:
                        idx = [idx[0]+1, 0, 1]
        return idx
    elif type == 'vgg':
        return idx
    elif type == 'darknet':
        return idx
def copy_cb(conv, bn, filter_idx, out=True):
    conv = copy_conv(conv, channel= filter_idx, out=out)   
    if bn:
        bn = copy_bn(bn, channel = filter_idx)  
        return conv, bn
    return conv

def block_remove(model, layer_idx, filter_idx):
    reslayer = [model.layer1, model.layer2, model.layer3, model.layer4]
    if layer_idx[0] == 0:  #濡傛灉鏄��涓�釜鍗风Н灞�        #灏嗘湰灞備腑鐨勫嵎绉�牳鍘绘帀
        model.conv1, model.bn1 = copy_cb(model.conv1, model.bn1, filter_idx, out=True)
        #鍘绘帀绗�竴灞備腑鐨勫嵎绉�牳
        nlayer = model.layer1[0]
        nlayer.conv1 = copy_conv(nlayer.conv1, channel=filter_idx, out=False)
        nlayer.downsample[0] = copy_conv(nlayer.downsample[0], channel=filter_idx, out=False)
    else:
        blayer = reslayer[layer_idx[0]-1][layer_idx[1]]
        if layer_idx[2] == 1:   #鏅��鐨勫嵎绉�            
            blayer.conv1, blayer.bn1 = copy_cb(blayer.conv1, blayer.bn1, filter_idx, out=True)
            blayer.conv2 = copy_conv(blayer.conv2, channel=filter_idx, out=False)
        elif layer_idx[2] == 2:
            blayer.conv2, blayer.bn2 = copy_cb(blayer.conv2, blayer.bn2, filter_idx, out=True)
            if blayer.conv3:    #鏅��鐨勫嵎绉�                
                blayer.conv3 = copy_conv(blayer.conv3, channel=filter_idx, out=False)                
            else:  
                if isinstance(blayer.downsample, nn.Sequential):    #绗�竴涓�潡
                    blayer.downsample[0], blayer.downsample[1] = copy_cb(blayer.downsample[0], blayer.downsample[1], filter_idx, out=True)
                else:                                               #鍏朵粬鐨勫潡
                    blayer.downsample = copy_conv(blayer.downsample, channel=filter_idx, out=True)  
                 
                blen = len(reslayer[layer_idx[0]-1])
                if layer_idx[1] < blen-1:
                    nlayer = reslayer[layer_idx[0]-1][layer_idx[1]+1]   #鑾峰彇涓嬩竴涓�潡
                    nlayer.conv1 = copy_conv(nlayer.conv1, channel=filter_idx, out=False)

                    nlayer.downsample = copy_conv(nlayer.downsample, channel=filter_idx, out=False)
                elif  layer_idx[1] == blen-1:                           #鑾峰彇涓嬩竴涓�潡
                    if layer_idx[0] != 3:
                        nlayer = reslayer[layer_idx[0]][0]
                        nlayer.conv1 = copy_conv(nlayer.conv1, channel=filter_idx, out=False)
                        nlayer.downsample[0] = copy_conv(nlayer.downsample[0], channel=filter_idx, out=False)
        else:  
            blayer.conv3, blayer.bn3 = copy_cb(blayer.conv3, blayer.bn3, filter_idx, out=True)  

            if isinstance(blayer.downsample, nn.Sequential):
                blayer.downsample[0], blayer.downsample[1] = copy_cb(blayer.downsample[0], blayer.downsample[1], filter_idx, out=True)
            else:
                blayer.downsample = copy_conv(blayer.downsample, channel=filter_idx, out=True) 
             
            blen = len(reslayer[layer_idx[0]-1])
            if layer_idx[1] < blen-1:
                nlayer = reslayer[layer_idx[0]-1][layer_idx[1]+1]       #鑾峰彇涓嬩竴涓�潡
                nlayer.conv1 = copy_conv(nlayer.conv1, channel=filter_idx, out=False)
                nlayer.downsample = copy_conv(nlayer.downsample, channel=filter_idx, out=False)
            elif  layer_idx[1] == blen-1:
                if layer_idx[0] != 4:
                    nlayer = reslayer[layer_idx[0]][0]                  #鑾峰彇涓嬩竴涓�潡
                    nlayer.conv1 = copy_conv(nlayer.conv1, channel=filter_idx, out=False)
                    nlayer.downsample[0] = copy_conv(nlayer.downsample[0], channel=filter_idx, out=False)
    return model


def filter_remove(model, layer_idx, filter_idx):
    bn_exist = False
    layer_len = len(model.features._modules.items())
    item = list(model.features._modules.items())
    cd = layer_idx
    _, conv_cur = item[layer_idx]
    #conv_cur = self.model.features[layer_idx] # self.model.features._modules.items()
    conv_next = None
    offset = 1
    ## locate the position of next conv and bitchnorm if exist
    while layer_idx+offset < layer_len:
        name_conv, res =item[layer_idx+offset]
        if isinstance(res, nn.Conv2d):
            conv_next = res
            break
        elif isinstance(res, nn.BatchNorm2d):
            bn_next = res
            bn_idx = offset
            bn_exist = True
        offset = offset + 1

    conv_new = copy_conv(conv_cur, channel= filter_idx, out=True)   
    if bn_exist: bn_new = copy_bn(bn_next, channel = filter_idx)
    if conv_next:
        conv_next_new = copy_conv(conv_next, channel=filter_idx, out=False)
        #print(conv_next_new)

    ##replace layer
    if conv_next:
        if bn_exist: 
            features = replace_Sequential(model.features, \
                                          [layer_idx, layer_idx+bn_idx, layer_idx+offset], \
                                          [conv_new, bn_new, conv_next_new])
        else:
            features = replace_Sequential(model.features, \
                                          [layer_idx, layer_idx+offset], \
                                          [conv_new, conv_next_new])                
    else:
        if bn_exist: 
            features = replace_Sequential(model.features, \
                                          [layer_idx, layer_idx+bn_idx], [conv_new, bn_new])
        else:
            features = replace_Sequential(model.features, \
                                          [layer_idx], [conv_new])                       
    #print(features)
    del model.features
    model.features = features
    return model

def log_info(f, str):
    f.write(str)
    f.flush()
class AutoPruner(object):
    def __init__(self, args, model, datasets, criterion, optimizer):
        self.args = args
        self.model = model
        self.datasets = datasets
        self.criterion = criterion
        self.optimizer = optimizer
        self.kd_model = None
        #self.kd_loss = 
        #self.train_top1 = AverageMeter()
        #self.train_top5 = AverageMeter()
        #self.train_losses = AverageMeter()
        #self.valid_top1 = AverageMeter()
        #self.valid_top5 = AverageMeter()
        #self.valid_losses = AverageMeter()        
        #self.batch_time = AverageMeter()
        #self.data_time = AverageMeter()  
        self.best_prec = 0
        self.classes = 10
        self.conv_cur = 0
        self.layer_cur = 0
        self.init_layer = 0
        self.last_conv = 40   #24  40
        if self.type == 'resnet':
            self.model_len = 15 * len(model.layer3)
        elif self.type == 'darknet':
            self.model_len = len(model.features) 
        self.alpha = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.beta = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.param = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.PARAM = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.CH = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.ch = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.ch_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.ch_log = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.ch_sqrt = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.cal_mean = torch.tensor(0, dtype = torch.float).to(device)
        self.cal_std = torch.tensor(0, dtype = torch.float).to(device)   
        self.channel = 0
        self.rchannel = 0
        self.acc_train_top1 = 0
        self.acc_train_top5 = 0
        self.acc_valid_top1 = 0
        self.acc_valid_top5 = 0
        self.kd_temp = 1.0
        self.kd_dw = 0.7
        self.kd_sw = 1 - self.kd_dw
        self.kd_tw = 0.
        self.g_aw = 0.3
        #self.acc_d = 0
        #self.acc_d_ = 0
        
        self.reset()
    
    def reset(self):
        conv_idx = 0;
        for idx, layer in enumerate(self.model.features.children()):
            if isinstance(layer, nn.Conv2d):
                s0, s1, s2, s3 = layer.weight.size()
                self.param[idx], self.ch[idx], conv_idx = s0 * s1 * s2 *s3, s0, idx
                
            if isinstance(layer, nn.ReLU):
                self.ch[idx], self.ch[conv_idx] = self.ch[conv_idx], 0
        #self.ch_, self.ch_log, self.ch_sqrt = self.ch, self.ch.log(), self.ch.sqrt()
        #self.ch_log = self.ch.log()
        #self.ch_sqrt = self.ch.sqrt()
        #self.ch_ = self.ch
                      
    
    def policy(self):
        mean = [0, 0, 0.14, 0, 
                0, 0, 0.15, 0,
                0, 0, 0.16, 0,
                0, 0, 0.13, 0,
                0, 0, 0.18, 0,
                0, 0, 0.29, 0,
                0, 0, 0.27, 0,
                0, 0]
        std = [0, 0, 0.18, 0,
               0, 0, 0.2, 0,
                0, 0, 0.24, 0,
                0, 0, 0.28, 0,
                0, 0, 0.29, 0,
                0, 0, 0.33, 0,
                0, 0, 0.22, 0,
                0, 0]               
        if self.alpha[self.layer_cur] < mean[self.layer_cur]: self.alpha[self.layer_cur] += 0.01
        if self.beta[self.layer_cur] < std[self.layer_cur]: self.beta[self.layer_cur] += 0.01
        
    
    def train(self, msg = False):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter() 
        
        ## switch to train mode
        self.model.train()
        
        tic = time.time()
        for i, (input, target) in enumerate(self.datasets['train']):
            # = target.cuda(async=True)
            input, target = input.to(device), target.to(device)          
            # compute output
            output = self.model(input)
            loss = self.criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))     
            
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   
            
        if msg:
            print('Train-Time {0:.3f}\tLoss {1:.4f}\tPrec@1 {2:.3f}\tPrec@5 {3:.3f} '.format(
              time.time()-tic, losses.avg, top1.avg, top5.avg))       
        log_info(ftrain, '[+] train {0:.4f}  {1:.4f} {2:.4f} {3:.4f} \n'.format(
                time.time()-tic, losses.avg, top1.avg, top5.avg))        
        return top1.avg, top5.avg, losses.avg
    
    def kd_loss(self, output, target, toutput):
        soft_log_probs = F.log_softmax(output/self.kd_temp, dim=1)
        soft_targets = F.softmax(toutput/self.kd_temp, dim=1)
        distill_loss = nn.KLDivLoss()(soft_log_probs, soft_targets)
        loss = distill_loss * (self.kd_temp*self.kd_temp*self.kd_sw) + F.cross_entropy(output, target) * (self.kd_dw)
        return loss.to(device)
    
    def train_kd(self, tmodel, msg = False):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter() 
        
        ## switch to train mode
        self.model.train()
        tmodel.eval()
        
        tic = time.time()
        for i, (input, target) in enumerate(self.datasets['train']):
            # = target.cuda(async=True)
            input, target = input.to(device), target.to(device)          
            # compute output
            output = self.model(input)
            toutput = tmodel(input).detach()
            
            loss = self.kd_loss(output, target, toutput)
            #loss = self.criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))     
            
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   
            
        if msg:
            print('Train-Time {0:.3f}\tLoss {1:.4f}\tPrec@1 {2:.3f}\tPrec@5 {3:.3f} '.format(
              time.time()-tic, losses.avg, top1.avg, top5.avg))  
        log_info(ftrain, '[*] train_kd {0:.4f}  {1:.4f} {2:.4f} {3:.4f} \n'.format(
                time.time()-tic, losses.avg, top1.avg, top5.avg))           
        return top1.avg, top5.avg, losses.avg
    
    def train_epochs(self, tmodel = None, epochs=1, pitch = 20, start_epoch=0, msg = True, store = False):
        self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5 = 0, 0, 0, 0
        for epoch in range(start_epoch, epochs):
            adjust_learning_rate(self.args, self.optimizer, epoch, pitch=pitch)
            tic = time.time()
            
            if tmodel:
                train_top1, train_top5, train_losses = self.train_kd(tmodel = tmodel)
            else:
                train_top1, train_top5, train_losses = self.train()
            if msg: print('Train[{4}]-Time {0:.3f}\tLoss {1:.4f}\tPrec@1 {2:.3f}\tPrec@5 {3:.3f} '
                      .format(time.time()-tic, train_losses, train_top1, train_top5, epoch))                             
            
            if train_top1 > self.acc_train_top1:
            #if train_top5 >= self.acc_train_top5:
                self.acc_train_top1, self.acc_train_top5 = train_top1, train_top5
            
            ##涓��婊ゆ尝鏂癸拷?            #if epoch > 0:
                #self.acc_train_top1 = 0.9 * self.acc_train_top1 + 0.1 * train_top1
                #self.acc_train_top5 = 0.9 * self.acc_train_top5 + 0.1 * train_top5
            #else:
                #self.acc_train_top1, self.acc_train_top5 = train_top1, train_top5
            
            toc = time.time()
            valid_top1, valid_top5, valid_losses = self.valid()
            if msg: print('Valid[{4}]-Time {0:.3f}\tLoss {1:.4f}\tPrec@1 {2:.3f}\tPrec@5 {3:.3f} '
                      .format(time.time()-toc, valid_losses, valid_top1, valid_top5, epoch))                        
            
            if valid_top1 > self.acc_valid_top1:
            #if valid_top5 >= self.acc_valid_top5:
                self.acc_valid_top1 = valid_top1
                self.acc_valid_top5 = valid_top5
            
            #if epoch > 0:
                #self.acc_valid_top1 = 0.9 * self.acc_valid_top1 + 0.1 * valid_top1
                #self.acc_valid_top5 = 0.9 * self.acc_valid_top5 + 0.1 * valid_top5
            #else:
                #self.acc_valid_top1, self.acc_valid_top5 = valid_top1, valid_top5            
        print('train:top1 {0:.3f} top5 {1:.3f}  valid:top1 {2:.3f} top5 {3:.3f}'.format(
            self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5))   
        log_info(ftrain, '[{0}]  {1:.4f}  {2:.4f} {3:.4f} {4:.4f} \n'.format(
                self.layer_cur, self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5))                
        if store:
            print('best_prec', self.best_prec)
            is_best = valid_top1 > self.best_prec #best_prec = max(prec1, best_prec)
            save_checkpoint(self.model, is_best, args.path)        
            self.best_prec = max(valid_top1, self.best_prec)            

    
    def valid(self, msg = False):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # switch to evaluate mode
        self.model.eval()
        tic = time.time()
        for i, (input, target) in enumerate(self.datasets['val']):
            input, target = input.to(device), target.to(device)
            # compute output
            output = self.model(input)
            loss = self.criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0)) 
        if msg:
            print('Valid-Time {0:.3f}\tLoss {1:.4f}\tPrec@1 {2:.3f}\tPrec@5 {3:.3f} '.format(
              time.time()-tic, losses.avg, top1.avg, top5.avg))    
        log_info(ftrain, 'valid {0:.4f}  {1:.4f} {2:.4f} {3:.4f} \n'.format(
                time.time()-tic, losses.avg, top1.avg, top5.avg))           
        return top1.avg, top5.avg, losses.avg
                  
    
    def run(self):
        pass
    
    #model, trainset, layer_cur = 0,  classes = 120, epochs = 0, diff = 0, 
    #parm={'alpha':1, 'beta':1, 'gamma':1, 'scale':1}
    def get_iter(self, acc):
        iter1, iter2 = 3, 2
        iter1 = min(max(math.ceil((-10 * acc)/ 2), iter1), 10)
        iter2 = min(max(math.ceil((-10 * acc) / 3), iter2), 10)
        return iter1, iter2    
    
    def get_mean_std(self, layer_cur = 0, msg = False):
        if msg: print('fileter selection begin--------------------')
        layer_find = None
        model = self.model
        self.cal_mean = 0
        self.cal_std = 0
                
        num = torch.tensor(0, dtype = torch.float).to(device)
    
        tic = time.time()
        layer_len = len(model.features._modules.items())
        for i, (xx, y) in enumerate(self.datasets['train']):
            xx, y = xx.to(device), y.to(device)
            for idx, layer in enumerate(model.features.children()):
                toc = time.time()
    
                # search for current convolution 
                if isinstance(layer, nn.Conv2d):
                    if not layer_find:  conv = idx
                    else:  
                        if idx < layer_cur: conv = idx
    
                if isinstance(layer, nn.ReLU):
                    ## initial cur_layer   the activation layer 
                    if not layer_find:
                        if layer_cur < idx :    
                            layer_cur = idx
                            layer_find = True
                    if idx == layer_cur:   # fine the current activation layer
                        x = model.features[:idx+1](xx).detach()
                        x_view = x.view(x.size(0), x.size(1),  -1)
                        x_mean = torch.mean(x_view, 2)
                        x_std = torch.std(x_view, 2) 
                        self.conv_cur = conv
                        self.layer_cur = layer_cur
                        break
                #print(time.time()-toc,'--', idx, ' : ', layer)
    
            ### deal with mean and std
            self.cal_mean = (num * self.cal_mean + y.size(0) * x_mean.mean(dim=0)) / (num + y.size(0))
            self.cal_std = (num * self.cal_std + y.size(0) * x_std.mean(dim=0)) / (num + y.size(0))
            num += y.size(0)

    
    def resnet_mean_std(self, layer_cur = [0,0,0], msg = False):
        if msg: print('fileter selection begin--------------------')
        model = self.model
        self.cal_mean = 0
        self.cal_std = 0
                
        num = torch.tensor(0, dtype = torch.float).to(device)
    
        tic = time.time()
        reslayer = [model.layer1, model.layer2, model.layer3, model.layer4]
        for i, (xx, _) in enumerate(self.datasets['train']):
            xx = xx.to(device)
            if layer_cur[0] == 0:
                xx = model.relu(model.bn1(model.conv1(xx)))
            else:
                xx = model.maxpool(model.relu(model.bn1(model.conv1(xx))))
                idx = layer_cur[0]-1
                for l in range(idx):
                    xx = reslayer[l](xx)
                xx = reslayer[idx][:layer_cur[1]](xx)
                blayer = reslayer[idx][layer_cur[1]]
                if layer_cur[2] == 1:
                    xx = nn.ReLU(inplace=True)(blayer.bn1(blayer.conv1(xx)))
                elif layer_cur[2] == 2:
                    if blayer.conv3:              
                        xx = nn.ReLU(inplace=True)(blayer.bn2(blayer.conv2(nn.ReLU(inplace=True)(blayer.bn1(blayer.conv1(xx))))))
                    else:
                        xx = blayer(xx)
                else:
                    xx = blayer(xx)
            x = xx.detach()  
            ## cal mean and std
            x_view = x.view(x.size(0), x.size(1),  -1)
            x_mean = torch.mean(x_view, 2)
            x_std = torch.std(x_view, 2)             
        
            ### deal with mean and std
            length = xx.size(0)
            self.cal_mean = (num * self.cal_mean + length * x_mean.mean(dim=0)) / (num + length)
            self.cal_std = (num * self.cal_std + length * x_std.mean(dim=0)) / (num +length)
            num += length
    
    def resnet_remove(self, channel):
        model = self.model
        reslayer = [model.layer1, model.layer2, model.layer3, model.layer4]
        layer_idx = self.getidx(self.layer_cur)
        if layer_idx[0] == 0:  #濡傛灉鏄��涓�釜鍗风Н灞�            #灏嗘湰灞備腑鐨勫嵎绉�牳鍘绘帀
            model.conv1, model.bn1 = copy_cb(model.conv1, model.bn1, channel, out=True)
            #鍘绘帀绗�竴灞備腑鐨勫嵎绉�牳
            nlayer = model.layer1[0]
            nlayer.conv1 = copy_conv(nlayer.conv1, channel=channel, out=False)
            nlayer.downsample[0] = copy_conv(nlayer.downsample[0], channel=channel, out=False)
        else:
            blayer = reslayer[layer_idx[0]-1][layer_idx[1]]
            if layer_idx[2] == 1:   #鏅��鐨勫嵎绉�                
                blayer.conv1, blayer.bn1 = copy_cb(blayer.conv1, blayer.bn1, channel, out=True)
                blayer.conv2 = copy_conv(blayer.conv2, channel=channel, out=False)
            elif layer_idx[2] == 2:
                blayer.conv2, blayer.bn2 = copy_cb(blayer.conv2, blayer.bn2, channel, out=True)
                if blayer.conv3:    #鏅��鐨勫嵎绉�                   
                    blayer.conv3 = copy_conv(blayer.conv3, channel=channel, out=False)                
                else:  
                    if isinstance(blayer.downsample, nn.Sequential):    #绗�竴涓�潡
                        blayer.downsample[0], blayer.downsample[1] = copy_cb(blayer.downsample[0], blayer.downsample[1], channel, out=True)
                    else:                                               #鍏朵粬鐨勫潡
                        blayer.downsample = copy_conv(blayer.downsample, channel=channel, out=True)  
                     
                    blen = len(reslayer[layer_idx[0]-1])
                    if layer_idx[1] < blen-1:
                        nlayer = reslayer[layer_idx[0]-1][layer_idx[1]+1]   #鑾峰彇涓嬩竴涓�潡
                        nlayer.conv1 = copy_conv(nlayer.conv1, channel=channel, out=False)

                        nlayer.downsample = copy_conv(nlayer.downsample, channel=channel, out=False)
                    elif  layer_idx[1] == blen-1:                           #鑾峰彇涓嬩竴涓�潡
                        if layer_idx[0] != 3:
                            nlayer = reslayer[layer_idx[0]][0]
                            nlayer.conv1 = copy_conv(nlayer.conv1, channel=channel, out=False)
                            nlayer.downsample[0] = copy_conv(nlayer.downsample[0], channel=channel, out=False)
            else:  
                blayer.conv3, blayer.bn3 = copy_cb(blayer.conv3, blayer.bn3, channel, out=True)  

                if isinstance(blayer.downsample, nn.Sequential):
                    blayer.downsample[0], blayer.downsample[1] = copy_cb(blayer.downsample[0], blayer.downsample[1], channel, out=True)
                else:
                    blayer.downsample = copy_conv(blayer.downsample, channel=channel, out=True) 
                 
                blen = len(reslayer[layer_idx[0]-1])
                if layer_idx[1] < blen-1:
                    nlayer = reslayer[layer_idx[0]-1][layer_idx[1]+1]       #鑾峰彇涓嬩竴涓�潡
                    nlayer.conv1 = copy_conv(nlayer.conv1, channel=channel, out=False)
                    nlayer.downsample = copy_conv(nlayer.downsample, channel=channel, out=False)
                elif  layer_idx[1] == blen-1:
                    if layer_idx[0] != 4:
                        nlayer = reslayer[layer_idx[0]][0]                  #鑾峰彇涓嬩竴涓�潡
                        nlayer.conv1 = copy_conv(nlayer.conv1, channel=channel, out=False)
                        nlayer.downsample[0] = copy_conv(nlayer.downsample[0], channel=channel, out=False)
                    else:
                        if model.fconv is not None:
                            model.fconv = copy_conv(model.fconv, channel=channel, out=False)
                        else:
                            pass
        return model






    def filter_remove(self, channel):
        model = self.model
        bn_exist = False
        item = list(model.features._modules.items())
        #cd = self.conv_cur
        _, conv_cur = item[self.conv_cur]
        #conv_cur = self.model.features[self.conv_cur] # self.model.features._modules.items()
        conv_next = None
        offset = 1
        ## locate the position of next conv and bitchnorm if exist
        while self.conv_cur+offset < self.model_len:
            name_conv, res =item[self.conv_cur+offset]
            if isinstance(res, nn.Conv2d):
                conv_next = res
                break
            elif isinstance(res, nn.BatchNorm2d):
                bn_next = res
                bn_idx = offset
                bn_exist = True
            offset = offset + 1
    
        conv_new = copy_conv(conv_cur, channel= channel, out=True)   
        if bn_exist: bn_new = copy_bn(bn_next, channel = channel)
        if conv_next:
            conv_next_new = copy_conv(conv_next, channel=channel, out=False)
            #print(conv_next_new)
    
        ##replace layer
        if conv_next:
            if bn_exist: 
                features = replace_Sequential(model.features, \
                                              [self.conv_cur, self.conv_cur+bn_idx, self.conv_cur+offset], \
                                              [conv_new, bn_new, conv_next_new])
            else:
                features = replace_Sequential(model.features, \
                                              [self.conv_cur, self.conv_cur+offset], \
                                              [conv_new, conv_next_new])                
        else:
            if bn_exist: 
                features = replace_Sequential(model.features, \
                                              [self.conv_cur, self.conv_cur+bn_idx], [conv_new, bn_new])
            else:
                features = replace_Sequential(model.features, \
                                              [self.conv_cur], [conv_new])                       
        #print(features)
        del model.features
        model.features = features
        return model        
    
    def get_goal(self):
        ## mean
        #print ('mean method - alpha', self.alpha[layer_cur])
        self.goal = 1- self.alpha[self.layer_cur] * (self.cal_mean + 0.0001).reciprocal()
        ## std
        #print ('std method - beta', self.beta[layer_cur])
        #self.goal = self.cal_std / self.beta[self.layer_cur] - 1
        ## std + mean
        #print('mean - ', self.alpha[layer_cur], ' and std - ', self.beta[layer_cur])
        #self.goal = self.cal_std / self.beta[self.layer_cur] - self.alpha[self.layer_cur] * (self.cal_mean + 0.0001).reciprocal()
        
    
    def model_select(self, model):
        pass
    
    def channel_remove(self, layer_cur=0, iters=1, iters_=3, msg=True):
        if msg: print('Prune channel ...', end=' ')
        self.get_mean_std(layer_cur)    
        if msg: print('\rPC: get mean&std', end=' ')
        self.policy()
        if msg: print('--> (s to a)', end=' ')
        self.get_goal()
        if msg: print('--> cal goal', end=' ')
        self.rchannel, filter_idx = self.goal.sort()
        if msg: print('--> sort importance')
        print('\n[{0}]: alpha {1:.4f} --> {2:.4f} a: {3:.5f}'.format(
            self.layer_cur, self.alpha_[self.layer_cur], self.alpha[self.layer_cur], self.a1[self.layer_cur]))
        if log: log_info(fp, '{0} {1:.4f} {2:.4f} {3:.5f}\n'.format(
            self.layer_cur, self.alpha_[self.layer_cur], self.alpha[self.layer_cur], self.a1[self.layer_cur]))
        #if log: fp.write('{0} {1:.4f} {2:.4f} {3:.5f}\n'.format(
            #self.layer_cur, self.alpha_[self.layer_cur], self.alpha[self.layer_cur], self.a1[self.layer_cur]))
        #if log: fp.flush()
        #print('train the model ...', end='\r')
        
        if self.a1[self.layer_cur] < self.a_base * 0.1:
            self.train_epochs(epochs=3, tmodel = self.kd_model, msg=False)
            return self.model
                    
        if self.rchannel[0] < 0 and (self.rchannel<0).sum() < self.rchannel.size(0):
            channel =  filter_idx.masked_select(self.rchannel.le(0)) 
            model_ = copy.deepcopy(self.model)
            self.ch_[self.layer_cur] = (self.rchannel>0).sum()
            self.model = self.filter_remove(channel)
            
            torch.cuda.empty_cache()
            
            ##  add optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.prune_lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay) 
            
            print(self.model.features[self.conv_cur])
            if msg: print('weak Ch:', self.rchannel[:(self.rchannel<0).sum()], 'Total Ch:', (self.rchannel<0).sum(),' / ', self.rchannel.size(0))
            self.train_epochs(epochs=iters)
            return model_
        else: 
            self.train_epochs(epochs=iters_, msg=False)
            return self.model
        
    def single_layer_prune(self):
        self.train_epochs(epochs=1, msg=True)
        acc_baseline = self.acc_valid_top5
        iters = 0
        prune_loop = True
        while prune_loop:
            self.train_epochs(epochs=3)
            _ = self.channel_remove(layer_cur=self.layer_cur, iters=1)
            
            iters += 1
            if iters == 20:
                iters = 0
                self.layer_cur = self.layer_cur+1
                if self.conv_cur == self.last_conv:
                    prune_loop = False
            else:
                self.layer_cur = self.conv_cur
                
    def whole_layer_prune(self):
        self.train_epochs(epochs=1, msg=True)
        acc_baseline = self.acc_valid_top5
        prune_loop = True
        iters = 0
        while prune_loop:
            self.train_epochs(epochs=1)
            #self.acc_d_ = self.acc_valid_top5 - acc_baseline
            
            _ = self.channel_remove(layer_cur=self.layer_cur, iters=1)
            
            self.layer_cur = self.layer_cur+1
            if self.conv_cur == self.last_conv:
                self.layer_cur = 0
                iters += 1
                if iters == 10:
                    prune_loop = False
                
                
    
    def lala(self):
        pass
    
class FBpruner(AutoPruner):
    def __init__(self, args, model, datasets, criterion, optimizer):
        #super(FBpruner, self).__init__(args, model, datasets, criterion, optimizer)
        self.model_len = len(model.features)
        self.acc_d = 0
        self.acc_d_ = 0  
        self.a_base = 0.01
        self.b_base = 0.01
        self.alpha_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.beta_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.a1 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.a2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b1 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        #self.reset()
        super(FBpruner, self).__init__(args, model, datasets, criterion, optimizer)
        
    def reset(self):
        conv_idx = 0;
        for idx, layer in enumerate(self.model.features.children()):
            if isinstance(layer, nn.Conv2d):
                s0, s1, s2, s3 = layer.weight.size()
                self.param[idx], self.ch[idx], conv_idx = s0 * s1 * s2 *s3, s0, idx

            if isinstance(layer, nn.ReLU):
                self.ch[idx], self.ch[conv_idx] = self.ch[conv_idx], 0
                self.a1[idx], self.a2[idx] = self.a_base, 20
                self.b1[idx], self.b2[idx] = self.b_base, 20
        self.ch_, self.ch_log, self.ch_sqrt = self.ch, self.ch.log(), self.ch.sqrt()
        self.ch[self.model_len-2]=0
        #self.ch_log = self.ch.log()
        #self.ch_sqrt = self.ch.sqrt()    
    
    
    def policy(self):
            self.alpha_[self.layer_cur] = self.alpha[self.layer_cur]
            #绛栫暐2
            #if method == 'mean':
            self.alpha[self.layer_cur] += self.a1[self.layer_cur] * \
                self.ch[self.layer_cur].log() / (self.ch_log[self.layer_cur])            
    
    def single_layer_prune(self):
        
        print('prune single layer with facebace')
        self.train_epochs(epochs=5, msg=False)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        prune_loop = True
        err_count, weak, a = 0, 0, 1
        self.acc_d_ = self.acc_valid_top5 - acc_baseline
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            print('after prune {0:.4f}\n'.format( self.acc_valid_top5 - acc_baseline))
            
            #鏍规嵁绮惧害鏇存柊鍙傛暟
            acc_d_ = self.acc_valid_top5 - acc_baseline
            if acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                a = (1 + 0.5 * acc_d_) if acc_d_ > 0 else (1 + acc_d_)
                if acc_d_ < -0.2:   #杈冨樊鐨勬�鑳�                    weak += 1
                    if weak < 2:    #鍏佽�鐨勮�瀵熸湡锟�                        self.model = model
                        self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                        a /= 1 + weak
                    else:           #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                        weak, a = 0, 0
                else:   
                    weak , err_count = 0, 0
                   
                #print ('a = {0:.4f}'.format(a))
                if a == 0:          #瓒呭嚭棰勬湡鐨勬儏鍐碉紝閲嶆柊瀹氫箟姝ラ暱
                    self.a1[self.layer_cur] = self.a_base
                    #print('绯绘暟', self.a1[self.layer_cur])
                    self.layer_cur = self.conv_cur
                    #prune_loop = False 
                    continue
                else:               #
                    self.a1[self.layer_cur] = a * self.a_base
                
                #淇濈暀濂界殑閫氶亾鏁伴噺
                self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                ##灏嗗綋鍓嶇殑绮惧害杩涜�澶囦唤
                #self.acc_d = self.acc_d_
                self.layer_cur = self.conv_cur
                
            else:
                err_count += 1
                if err_count == 3:
                    print(self.model)
                    self.model = torch.load('checkpoint/20/model').to(device)
                    self.layer_cur = self.layer_cur+1
                    err_count = 0
                    if self.conv_cur == self.last_conv:
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
        print('Pruned finish ! ...')

    def single_layer_fastprune(self):
        print('prune single layer with facebace')
        self.train_epochs(epochs=5, msg=False)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        prune_loop = True
        err_count, weak, a = 0, 0, 1
        acc_line = 0
        self.acc_d_ = self.acc_valid_top5 - acc_baseline
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            
            #鏍规嵁绮惧害鏇存柊鍙傛暟
            self.acc_d_ = self.acc_valid_top5 - acc_baseline
            acc_line = 0.9 * acc_line + 0.1 * self.acc_d_
            print('Prune_acc {0:.4f} baseline {1:.4f}\n'.format(self.acc_d_, acc_line))
            if self.acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                a = (1 + 0.5 * self.acc_d_) if self.acc_d_ > 0 else (1 + self.acc_d_)
                
                if (self.rchannel<0).sum() > 0:
                    if self.acc_d_ < acc_line:   #杈冨樊鐨勬�鑳�                        weak += 1
                        if weak < 3:    #鍏佽�鐨勮�瀵熸湡锟�                            self.model = model
                            self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                            a /= 1 + weak
                        else:           #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                            weak, a = 0, 0
                    else:   
                        weak , err_count = 0, 0
                       
                    if a == 0:          #瓒呭嚭棰勬湡鐨勬儏鍐碉紝閲嶆柊瀹氫箟姝ラ暱
                        self.a1[self.layer_cur] = self.a_base
                        self.layer_cur = self.conv_cur
                        prune_loop = False 
                        continue

                self.a1[self.layer_cur] = a * self.a_base
                
                #淇濈暀濂界殑閫氶亾鏁伴噺
                self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                self.layer_cur = self.conv_cur
                
            else:
                err_count += 1
                if err_count == 3:
                    print(self.model)
                    self.model = torch.load('checkpoint/20/model').to(device)
                    self.layer_cur = self.layer_cur+1
                    acc_line = 0
                    err_count = 0
                    if self.conv_cur == self.last_conv:
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
        print('Pruned finish ! ...')

    def whole_layer_prune(self):
        print('prune whole layer with facebace')
        self.train_epochs(epochs=5, msg=True)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )        
        prune_loop = True
        #iters = 0
        err_count = torch.zeros(self.model_len)
        weak = [0] * self.model_len #torch.zeros(self.model_len)
        a = 1
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            print('after prune {0:.4f}\n'.format( self.acc_valid_top5 - acc_baseline))
            
            #鏍规嵁绮惧害鏇存柊鍙傛暟
            self.acc_d_ = self.acc_valid_top5 - acc_baseline
            if self.acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                a = (1 + 0.5 * self.acc_d_) if self.acc_d_ > 0 else (1 + self.acc_d_)
                if self.acc_d_ < -0.2:   #杈冨樊鐨勬�鑳�                    weak[self.layer_cur] += 1
                    if weak[self.layer_cur] < 2:    #鍏佽�鐨勮�瀵熸湡锟�                        self.model = model
                        self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                        a /= 1 + weak[self.layer_cur]
                    else:           #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                        weak[self.layer_cur], a = 0, 0
                else:   
                    weak[self.layer_cur] , err_count[self.layer_cur] = 0, 0 
                    
                if a == 0:          #瓒呭嚭棰勬湡鐨勬儏鍐碉紝閲嶆柊瀹氫箟姝ラ暱
                    self.a1[self.layer_cur] = self.a_base
                    #print('绯绘暟', self.a1[self.layer_cur])
                    self.layer_cur += 1
                    if self.conv_cur == self.last_conv:
                        self.layer_cur = 0                    
                    #prune_loop = False 
                    continue
                else:               #
                    self.a1[self.layer_cur] = a * self.a_base
                
                #淇濈暀濂界殑閫氶亾鏁伴噺
                self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0   
            else:
                err_count[self.layer_cur] += 1
                if err_count[self.layer_cur] == 3:
                    self.layer_cur = self.layer_cur+1
                    if self.conv_cur == self.last_conv:
                        print(self.model)
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0
        print('Pruned finish ! ...')
  
    def whole_layer_fastprune(self):
        print('prune whole layer with facebace')
        self.train_epochs(epochs=5, msg=True)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )        
        prune_loop = True
        
        err_count = (self.ch == 0) * 10 #err_count = torch.zeros(self.model_len)
        acc_line = (self.ch > 0).float() -1
        
        weak = [0] * self.model_len #torch.zeros(self.model_len)
        a, acc, lamda, err_ = 1, 0, 0.95, 1.
        
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            
            #鏍规嵁绮惧害鏇存柊鍙傛暟
            self.acc_d_ = self.acc_valid_top5 - acc_baseline
            acc_line[self.layer_cur] = 0.9 * acc_line[self.layer_cur] + 0.1 * self.acc_d_
            print('Prune_acc {0:.4f} baseline {1:.4f}\n'.format(self.acc_d_, acc_line[self.layer_cur]))
            if log: log_info(facc, '{0} {1:.4f} {2:.4f} {3:.4f}\n' .format(
                self.layer_cur, self.acc_d_, acc_line[self.layer_cur], max(acc_line) * self.g_aw + (1-self.g_aw) * acc_line[self.layer_cur]))
            self.a1[self.layer_cur].clamp(min=0.1*self.a_base, max=self.a_base)
            if self.acc_d_ > err_:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                #acc_line[self.layer_cur] = lamda * acc_line[self.layer_cur] + (1-lamda) * err_
                acc = max(acc_line) * self.g_aw + (1-self.g_aw) * acc_line[self.layer_cur]
                a = (1 + 0.5 * self.acc_d_) if self.acc_d_ > 0 else (1 + self.acc_d_)
                
                if (self.rchannel<0).sum() > 0:
                    if self.acc_d_ < acc:   #杈冨樊鐨勬�鑳�                        weak[self.layer_cur] += 1
                        self.model = model
                        self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                        a /= 1 + weak[self.layer_cur]
                    else:   
                        weak[self.layer_cur] , err_count[self.layer_cur] = 0, 0
                        
                self.a1[self.layer_cur] = a * self.a_base
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0   
            else:
                err_count[self.layer_cur] += 1
                if min(err_count) == 3:
                    print(self.model)
                    prune_loop = False     
                    break
                    #continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0
        print('Pruned finish ! ...')
class RLpruner(AutoPruner):
    def __init__(self, args, model, datasets, criterion, optimizer):
        self.model_len = len(model.features)
        self.acc_d = 0
        self.acc_d_ = 0  
        self.a_base = 0.01
        self.b_base = 0.01
        self.alpha_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.beta_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.a1 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.a2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b1 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        super(RLpruner, self).__init__(args, model, datasets, criterion, optimizer)
        
    def reset(self):
        conv_idx = 0;
        for idx, layer in enumerate(self.model.features.children()):
            if isinstance(layer, nn.Conv2d):
                s0, s1, s2, s3 = layer.weight.size()
                self.param[idx], self.ch[idx], conv_idx = s0 * s1 * s2 *s3, s0, idx

            if isinstance(layer, nn.ReLU):
                self.ch[idx], self.ch[conv_idx] = self.ch[conv_idx], 0
                self.a1[idx], self.a2[idx] = self.a_base, 20
                self.b1[idx], self.b2[idx] = self.b_base, 20
        self.ch[self.model_len-2]=0
        self.ch_, self.ch_log, self.ch_sqrt = self.ch.clone(), self.ch.log(), self.ch.sqrt()  
        
    
    def policy(self):
        self.alpha_[self.layer_cur] = self.alpha[self.layer_cur]
        self.alpha[self.layer_cur] += self.a1[self.layer_cur] * \
                self.ch[self.layer_cur].log() / (self.ch_log[self.layer_cur])            

    def single_layer_prune(self):
        print('prune single layer with reinforce')
        self.train_epochs(epochs=5, msg=False)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        prune_loop = True
        err_count, weak, a = 0, 0, 1
        acc_line = 0
        n = 1
        acc_d_ = (self.acc_valid_top5 - acc_baseline) #/ self.ch[self.layer_cur]        
        
        while prune_loop:
            iter1, iter2 = 3, 2#self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            
            #鏍规嵁绮惧害鏇存柊鍙傛暟
            self.acc_d_ = (self.acc_valid_top5 - acc_baseline) #/ self.ch_[self.layer_cur]
            #acc_line = 0.9 * acc_line + 0.1 * self.acc_d_
            print('Prune_acc {0:.4f} baseline {1:.4f}\n'.format(self.acc_d_, acc_d_))
            if log: log_info(facc, '{2} {0:.4f}  {1:.4f} {3:.4f} {4} {5}\n'.format(
                self.acc_d_, acc_d_, self.layer_cur, self.acc_d_/self.ch_[self.layer_cur] - acc_d_/self.ch[self.layer_cur], self.ch_[self.layer_cur], self.ch[self.layer_cur]))
            
            if self.acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                diff =  self.acc_d_/self.ch_[self.layer_cur] - acc_d_/self.ch[self.layer_cur] #self.acc_d_ - acc_d_
                a = (1 + diff) if diff > 0 else (1 + diff)
                
                if (self.rchannel<0).sum() > 0:
                    if diff > 0:                                       #姣旇緝濂界殑鐘跺喌
                        self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                        self.a1[self.layer_cur] = self.a_base * a
                        acc_d_ = self.acc_d_
                        err_count, weak, n = 0, 0, 1
                    else:                                                    #寰堥�鐨勬牱瀛�                        weak += 1
                        if weak < 3 :                                 #鍏佽�鐨勮�瀵熸湡锟�                            self.model = model
                            self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                            self.a1[self.layer_cur] = self.a_base * a / (1 + weak)
                        else:                                               #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                            weak = 0
                            self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                            self.a1[self.layer_cur] = self.a_base * a * 0.8
                            acc_d_ = self.acc_d_             
                            #n = 1
                else:                                                                               #娌℃湁鍔ㄤ綔鐨勬儏锟�                    self.a1[self.layer_cur] = self.a_base * a
                    acc_d_ = acc_d_ * 0.8 + self.acc_d_ * 0.2
                    #acc_d_ = (n * acc_d_ + self.acc_d_) / (n+1)
                    #n += 1
                    #acc_d_ = self.acc_d_
                self.layer_cur = self.conv_cur         
            else:
                err_count += 1
                if err_count == 3:
                    print(self.model)
                    self.model = torch.load('checkpoint/20/model').to(device)
                    self.layer_cur = self.layer_cur+1
                    err_count = 0
                    if self.conv_cur == self.last_conv:
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
                self.layer_cur = self.conv_cur
        print('Pruned finish ! ...')
        
    def whole_layer_prune(self):
        print('prune whole layer with reinforce')
        self.train_epochs(epochs=5, msg=False) #5
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        prune_loop = True
        a, lamda, err_, acc, MAXweak = 1, 0.95, -1, 0, 4
        err_count = (self.ch == 0) * 10
        acc_d_ = (rlmodel.ch > 0).float() * 2 * (-err_) + err_
        weak = [1] * self.model_len #torch.zeros(self.model_len)
        
        self.kd_model = copy.deepcopy(self.model)
        '''
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline) #3, 2
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            
            #鏍规嵁绮惧害鏇存柊鍙傛暟
            self.acc_d_ = self.acc_valid_top5 - acc_baseline - err_
            print('Prune_acc {0:.4f} self.acc_d {1:.4f} diff {2:.4f} acc_line {3:.4f}\n'.format(self.acc_d_, acc_d_[self.layer_cur], self.acc_d_/self.ch_[self.layer_cur] - acc_d_[self.layer_cur]/self.ch[self.layer_cur], max(acc_d_)))
            if log: log_info(facc, '{2} {0:.4f}  {1:.4f} {3:.4f} {4:.0f} {5:.0f}\n'.format(
                self.acc_d_, acc_d_[self.layer_cur], self.layer_cur, self.acc_d_/self.ch_[self.layer_cur] - max(acc_d_)/self.ch[self.layer_cur], self.ch_[self.layer_cur], self.ch[self.layer_cur]))
            
            #self.a1[self.layer_cur].clamp(min=0.1*self.a_base, max=self.a_base)
            if self.acc_d_ > 0:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟� self.acc_valid_top5 - acc_baseline
                acc_d_[self.layer_cur] = lamda * acc_d_[self.layer_cur] #+ (1-lamda) * err_
                acc = max(acc_d_) * self.g_aw + (1 - self.g_aw) * acc_d_[self.layer_cur]
                diff = self.acc_d_/self.ch_[self.layer_cur] - acc/self.ch[self.layer_cur]
                a = (1 + diff) if diff > 0 else (1 + 2 * diff)
                print('[{0}] acc_d_ {1:.5f}/{2:.5f} acc {3:.5f} diff {4:.5f}' .format(self.layer_cur, acc_d_[self.layer_cur], max(acc_d_), acc, diff))    
                
                if (self.rchannel<0).sum() > 0:                             #浜х敓浜嗕慨鍓�殑锟�                    if diff > 0:                                       #姣旇緝濂界殑鐘跺喌
                        self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                        weak[self.layer_cur] = 1 if weak[self.layer_cur] == 1 else weak[self.layer_cur] - 1
                        acc_d_[self.layer_cur] = self.acc_d_
                        err_count[self.layer_cur] = 0
                    else:                                                                             #寰堥�鐨勬牱瀛�                        #weak[self.layer_cur] += 1
                        weak[self.layer_cur] = MAXweak if weak[self.layer_cur] > MAXweak-1 else weak[self.layer_cur] + 1
                        self.model = model
                        self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                        #self.a1[self.layer_cur] /= (1 + weak[self.layer_cur])  
                        acc_d_[self.layer_cur] = acc_d_[self.layer_cur] * 0.9 + self.acc_d_ * 0.1
                else:                                                                               #娌℃湁鍔ㄤ綔鐨勬儏锟�                    #self.a1[self.layer_cur] *= 1
                    self.alpha[self.layer_cur] = max(self.alpha[self.layer_cur], 
                                                     min(self.cal_mean) - 0.5* self.a1[self.layer_cur])
                    acc_d_[self.layer_cur] = acc_d_[self.layer_cur] * 0.8 + self.acc_d_ * 0.2
                
                self.a1[self.layer_cur] = self.a_base *  a / weak[self.layer_cur]  

            else:
                err_count[self.layer_cur] += 1
                print ('err:', min(err_count), err_count)
                if min(err_count) == 3:
                    print(self.model)
                    prune_loop = False     
                    break
                    #continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
                
            self.layer_cur += 1
            if self.conv_cur == self.last_conv:
                self.layer_cur = 0                
        print('Pruned finish ! ...') '''
                

class RLpruner2(AutoPruner):
    def __init__(self, args, model, datasets, criterion, optimizer):
        self.type = 'resnet'
        if self.type == 'darknet':
            self.reslen = 6
            self.model_len = len(model.features) 
        elif self.type == 'resnet':
            self.reslen = len(model.layer3)
            self.model_len = 15 * self.reslen
        
        #self.reslen = 6 if model.layer3 is None else len(model.layer3)
        #self.model_len = len(model.features) if model.features is not None else 15 * self.reslen
        self.acc_d = 0
        self.acc_d_ = 0  
        self.a_base = 0.01
        self.b_base = 0.01
        self.alpha_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.beta_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.a1 = torch.zeros(self.model_len, dtype = torch.int).to(device)
        self.a2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b1 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        
        super(RLpruner2, self).__init__(args, model, datasets, criterion, optimizer)
        
    def reset(self):
        if self.type == 'darknet':
            conv_idx = 0
            for idx, layer in enumerate(self.model.features.children()):
                if isinstance(layer, nn.Conv2d):
                    s0, s1, s2, s3 = layer.weight.size()
                    self.param[idx], self.ch[idx], conv_idx = s0 * s1 * s2 *s3, s0, idx

                if isinstance(layer, nn.ReLU):
                    self.ch[idx], self.ch[conv_idx] = self.ch[conv_idx], 0
                    self.a1[idx], self.a2[idx] = (1*self.ch[idx].log()).int(), 0  #self.a_base
                    #self.b1[idx], self.b2[idx] = , 20 #self.b_base
                    self.alpha[idx], self.beta[idx] = 1, 0
                    if self.init_layer == 0:
                        self.init_layer = idx
            self.ch[-1]=0
            self.CH, self.ch_, self.ch_log, self.ch_sqrt = self.ch.clone(), self.ch.clone(), self.ch.log(), self.ch.sqrt()  
            self.layer_cur = self.init_layer
        elif self.type == 'resnet':
            model = self.model
            reslayer = [model.layer1, model.layer2, model.layer3, model.layer4]
            for i in range(5):
                if i == 0:
                    s0, s1, s2, s3 = self.model.conv1.weight.size()
                    self.param[0], self.ch[0] = s0 * s1 * s2 *s3, s0
                    self.a1[0], self.a2[0] = (1*self.ch[0].log()).int(), 0
                    self.alpha[0], self.beta[0] = 1, 0
                else:
                    for j in range(len(reslayer[i-1])):
                        s0, s1, s2, s3 = reslayer[i-1][j].conv1.weight.size()
                        idx = self.getlayer([i,j,1])
                        self.param[idx], self.ch[idx] = s0 * s1 * s2 *s3, s0
                        self.a1[idx], self.a2[idx] = (1*self.ch[idx].log()).int(), 0
                        self.alpha[idx], self.beta[idx] = 1, 0

                        s0, s1, s2, s3 = reslayer[i-1][j].conv2.weight.size()
                        idx = self.getlayer([i,j,2])
                        self.param[idx], self.ch[idx] = s0 * s1 * s2 *s3, s0
                        self.a1[idx], self.a2[idx] = (1*self.ch[idx].log()).int(), 0
                        self.alpha[idx], self.beta[idx] = 1, 0
                        if reslayer[i-1][j].conv3 is not None:
                            s0, s1, s2, s3 = reslayer[i-1][j].conv3.weight.size()
                            idx = self.getlayer([i,j,0])
                            self.param[idx], self.ch[idx] = s0 * s1 * s2 *s3, s0
                            self.a1[idx], self.a2[idx] = (1*self.ch[idx].log()).int(), 0
                            self.alpha[idx], self.beta[idx] = 1, 0
            if reslayer[3][2].conv3 is not None: self.ch[self.getlayer([4,2,0])] = 0
            else:   self.ch[self.getlayer([4,2,2])] = 0
            self.CH, self.ch_, self.ch_log, self.ch_sqrt = self.ch.clone(), self.ch.clone(), self.ch.log(), self.ch.sqrt()
            self.layer_cur = self.getlayer([0,0,0])
     

    def getidx(self, layer):
        idx = [0, 0, 0]
        idx[2] = layer % 3
        layer = layer // 3
        idx[0], idx[1] = layer // self.reslen, layer % self.reslen
        return idx

    def getlayer(self, idx):
        return (idx[0] * self.reslen +idx[1])*3 + idx[2]

    def policy(self):
        self.alpha_[self.layer_cur] = self.alpha[self.layer_cur]
        self.alpha[self.layer_cur] += self.a1[self.layer_cur] * \
                self.ch[self.layer_cur].log() / (self.ch_log[self.layer_cur])            

    def get_goal(self, layer_cur):
        if self.type == 'darknet':
            self.get_mean_std(layer_cur)  
        elif self.type == 'resnet':
            self.resnet_mean_std(self.getidx(layer_cur))
        self.goal = -self.alpha[self.layer_cur] * (self.cal_mean + 0.0001).reciprocal()
        
    def channel_remove(self, layer_cur=0, iters=1, pitch = 3, iters_=3, msg=True):
        torch.cuda.empty_cache()
        if msg: print('Prune channel ...', end=' ')
        self.get_goal(layer_cur)
        if msg: print('--> cal goal', end=' ')
        self.rchannel, filter_idx = self.goal.sort()
        model_ = copy.deepcopy(self.model)
        print('\n[{0}]: a: {1} radio:{2:.3f}'.format(
            self.layer_cur, self.a1[self.layer_cur], self.a2[self.layer_cur]))
        
        channel = filter_idx[:self.a1[self.layer_cur]]
        self.ch_[self.layer_cur] = len(filter_idx[self.a1[self.layer_cur]:])
        if self.type == 'darknet':
            self.model = self.filter_remove(channel)
        elif self.type == 'resnet':
            self.model = self.resnet_remove(channel)
    
        torch.cuda.empty_cache()
    
        #  add optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.prune_lr,
                                                 momentum=self.args.momentum,
                                            weight_decay=self.args.weight_decay) 
        
        if self.type == 'darknet':  print(self.model.features[self.conv_cur])
        self.train_epochs(epochs=iters, pitch=pitch)
        return model_            
        
    def single_layer_prune(self):
        print('prune single layer with reinforce')
        self.train_epochs(epochs=5, msg=False)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        prune_loop = True
        err_count, weak, a = 0, 0, 1
        acc_line = 0
        n = 1
        acc_d_ = (self.acc_valid_top5 - acc_baseline) #/ self.ch[self.layer_cur]        
        
        while prune_loop:
            iter1, iter2 = 3, 2#self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            
            #鏍规嵁绮惧害鏇存柊鍙傛暟
            self.acc_d_ = (self.acc_valid_top5 - acc_baseline) #/ self.ch_[self.layer_cur]
            #acc_line = 0.9 * acc_line + 0.1 * self.acc_d_
            print('Prune_acc {0:.4f} baseline {1:.4f}\n'.format(self.acc_d_, acc_d_))
            if log: log_info(facc, '{2} {0:.4f}  {1:.4f} {3:.4f} {4} {5}\n'.format(
                self.acc_d_, acc_d_, self.layer_cur, self.acc_d_/self.ch_[self.layer_cur] - acc_d_/self.ch[self.layer_cur], self.ch_[self.layer_cur], self.ch[self.layer_cur]))
            
            if self.acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                diff =  self.acc_d_/self.ch_[self.layer_cur] - acc_d_/self.ch[self.layer_cur] #self.acc_d_ - acc_d_
                a = (1 + diff) if diff > 0 else (1 + diff)
                
                if (self.rchannel<0).sum() > 0:
                    if diff > 0:                                       #姣旇緝濂界殑鐘跺喌
                        self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                        self.a1[self.layer_cur] = self.a_base * a
                        acc_d_ = self.acc_d_
                        err_count, weak, n = 0, 0, 1
                    else:                                                    #寰堥�鐨勬牱瀛�                        weak += 1
                        if weak < 3 :                                 #鍏佽�鐨勮�瀵熸湡锟�                            self.model = model
                            self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                            self.a1[self.layer_cur] = self.a_base * a / (1 + weak)
                        else:                                               #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                            weak = 0
                            self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                            self.a1[self.layer_cur] = self.a_base * a * 0.8
                            acc_d_ = self.acc_d_             
                            #n = 1
                else:                                                                               #娌℃湁鍔ㄤ綔鐨勬儏锟�                    self.a1[self.layer_cur] = self.a_base * a
                    acc_d_ = acc_d_ * 0.8 + self.acc_d_ * 0.2
                    #acc_d_ = (n * acc_d_ + self.acc_d_) / (n+1)
                    #n += 1
                    #acc_d_ = self.acc_d_
                self.layer_cur = self.conv_cur         
            else:
                err_count += 1
                if err_count == 3:
                    print(self.model)
                    self.model = torch.load('checkpoint/20/model').to(device)
                    self.layer_cur = self.layer_cur+1
                    err_count = 0
                    if self.conv_cur == self.last_conv:
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
                self.layer_cur = self.conv_cur
        print('Pruned finish ! ...')
       
    def preprune(self):
        p = 0.2
        while(True):
            self.get_goal(self.layer_cur)
            self.rchannel, filter_idx = self.goal.sort()
            channel = filter_idx[:int(p*len(filter_idx))]
            self.ch_[self.layer_cur] = len(filter_idx[self.a1[self.layer_cur]:])
            self.ch[self.layer_cur] = self.ch_[self.layer_cur]
            self.a2[self.layer_cur] = p
            if self.type == 'darknet':
                self.model = self.filter_remove(channel)
            elif self.type == 'resnet':
                self.model = self.resnet_remove(channel)  
            self.train_epochs(tmodel=self.kd_model, epochs=5, msg=True, pitch=3)
            self.layer_cur = self.getlayer(get_nextlayer(self.model, self.getidx(self.layer_cur)))
            if self.layer_cur == 78:
                self.layer_cur = 0
                break
        self.train_epochs(tmodel=self.kd_model, epochs=30, msg=True, pitch=20)
        print('top-1 {0:.4f} top-5 {1:.4f}' .format(self.acc_valid_top1, self.acc_valid_top5))        
        
     
    def whole_layer_prune(self):
        print('prune whole layer with reinforce')
        self.train_epochs(epochs=5, msg=True) #5  86.49 #
        acc_baseline = self.acc_valid_top1 #86.825   #self.acc_valid_top5
        #print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        self.preprune()
        prune_loop = True
        a, lamda, elamda, err_, acc, MAXq, MAXerr = 1, 0.9, 0.8, -1, 0, 4, 3
        epoch = 0
        pitch = 3
        self.g_aw = 0.7
        err_count = (self.ch == 0) * 10
        acc_d_ = (self.ch > 0).float() * 2 * (-err_) + err_
        q = [1] * self.model_len #torch.zeros(self.model_len)
        
        self.kd_model = copy.deepcopy(self.model)
        while prune_loop:
            if err_count[self.layer_cur] > MAXerr:
                if max(acc_d_) == acc_d_[self.layer_cur]:
                    acc_d_[self.layer_cur] *= elamda
                
            else:
                iter1, iter2 = self.get_iter(self.acc_valid_top1 - acc_baseline) #3, 2
                model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2, pitch=pitch)
                
                #鏍规嵁绮惧害鏇存柊鍙傛暟
                self.acc_d_ = self.acc_valid_top1 - acc_baseline - err_
                print('{4} Prune_acc {0:.4f} self.acc_d {1:.4f} diff {2:.4f} acc_line {3:.4f}\n'.format(self.acc_d_, acc_d_[self.layer_cur], self.acc_d_/self.ch_[self.layer_cur] - acc_d_[self.layer_cur]/self.ch[self.layer_cur], max(acc_d_), epoch))
                if log: log_info(facc, '{6} {2} {0:.4f}  {1:.4f} {3:.4f} {4:.0f} {5:.0f}\n'.format(
                    self.acc_d_, acc_d_[self.layer_cur], self.layer_cur, self.acc_d_/self.ch_[self.layer_cur] - max(acc_d_)/self.ch[self.layer_cur], self.ch_[self.layer_cur], self.ch[self.layer_cur], epoch))
                
                if self.acc_d_ > 0:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                    acc_d_[self.layer_cur] = lamda * acc_d_[self.layer_cur] #+ (1-lamda) * err_
                    acc = max(acc_d_) * self.g_aw + (1 - self.g_aw) * acc_d_[self.layer_cur]
                    diff = self.acc_d_/self.ch_[self.layer_cur] - acc/self.ch[self.layer_cur]
                    a = (1 + diff) if diff > 0 else (1 + 2 * diff)
                    print('{5} [{0}] acc_d_ {1:.5f}/{2:.5f} acc {3:.5f} diff {4:.5f}' .format(self.layer_cur, acc_d_[self.layer_cur], max(acc_d_), acc, diff, epoch))    
                    Power = 1* self.ch[self.layer_cur].log() * a
                    #Power = self.Param[self.layer_cur].log()
                    
                    print('Policy {3:.4f} [{0}] q {1} a1 {2} '.format(self.layer_cur, q[self.layer_cur], self.a1[self.layer_cur], Power), end=' ') 
                    if diff > 0:
                        self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                        self.a2[self.layer_cur] += self.a1[self.layer_cur].float()/self.CH[self.layer_cur]
                        
                        if q[self.layer_cur] == 1:
                            print('g1', end='')
                            self.a1[self.layer_cur] += 1
                        elif q[self.layer_cur] == MAXq:
                            print('gm', end='')
                            if self.a1[self.layer_cur] <  (Power/q[self.layer_cur]).int():
                                print('<', end=' ')
                                self.a1[self.layer_cur] = (Power/q[self.layer_cur]).int()
                            else:
                                print('>=', end=' ')
                                q[self.layer_cur] -= 1
                                self.a1[self.layer_cur] = (Power/q[self.layer_cur]).int()                                
                        else:
                            print('g-', end='')
                            q[self.layer_cur] = 1 if q[self.layer_cur]==1 else q[self.layer_cur]-1
                            self.a1[self.layer_cur] = (Power/q[self.layer_cur]).int()
                        if self.a1[self.layer_cur] >= self.ch[self.layer_cur].int():
                            self.a1[self.layer_cur] = self.ch[self.layer_cur] - 1
                        acc_d_[self.layer_cur] = self.acc_d_
                        err_count[self.layer_cur] = 0  
                    else:
                        self.model = model
                        if q[self.layer_cur] == 1:
                            print('b1', end='')
                            if self.a1[self.layer_cur] > Power.int():
                                print('>', end=' ')
                                self.a1[self.layer_cur] = Power.int()
                            else:
                                print('<=', end=' ')
                                q[self.layer_cur] += 1
                                self.a1[self.layer_cur] = (Power/q[self.layer_cur]).int()                                
                        elif q[self.layer_cur] == MAXq:
                            print('bm', end='')
                            if self.a1[self.layer_cur] >  1:    #(Power/q[self.layer_cur]).int():
                                print('>', end=' ')
                                self.a1[self.layer_cur] -= 1                       
                        else:
                            print('b-', end='')
                            q[self.layer_cur] += 1
                            self.a1[self.layer_cur] = (Power/q[self.layer_cur]).int()
                        if self.a1[self.layer_cur] <= 1:
                            print('ba', end=' ')
                            self.a1[self.layer_cur] = 1
                            q[self.layer_cur] = 1 if q[self.layer_cur] == 1 else q[self.layer_cur]-1
                        acc_d_[self.layer_cur] = acc_d_[self.layer_cur] * 0.9 + self.acc_d_ * 0.1
                    print('-> q {0} a1 {1}'.format(q[self.layer_cur], self.a1[self.layer_cur])) 
                else:
                    if max(acc_d_) == acc_d_[self.layer_cur]:
                        acc_d_[self.layer_cur] *= elamda
                    err_acc = self.acc_valid_top5
                    err_count[self.layer_cur] += 1
                    print ('err:', min(err_count), err_count)                    
                    self.model = model

                    if q[self.layer_cur] == MAXq:
                        if self.a1[self.layer_cur] >  1:    #(Power/q[self.layer_cur]).int():
                            self.a1[self.layer_cur] -= 1                       
                    else:
                        q[self.layer_cur] += 1
                        self.a1[self.layer_cur] = (self.a1[self.layer_cur] *(q[self.layer_cur]-1) /q[self.layer_cur]).int()
                    if self.a1[self.layer_cur] <= 1:
                        self.a1[self.layer_cur] = 1
                        #q[self.layer_cur] = 1 if q[self.layer_cur] == 1 else q[self.layer_cur]-1
                    #acc_d_[self.layer_cur] = acc_d_[self.layer_cur] * 0.9 + self.acc_d_ * 0.1
                    #print('-> q {0} a1 {1}'.format(q[self.layer_cur], self.a1[self.layer_cur]))                     
                    
                    self.train_epochs(tmodel=self.kd_model, epochs=5, msg=True)
                    print('[{0}] pre_acc {1:.4f} kd_acc {2:.4f}' .format(self.layer_cur, err_acc, self.acc_valid_top5))
                    if min(err_count) == MAXerr:
                        print(self.model)  
                        break 
            if self.type == 'darknet':                   
                self.layer_cur += 1
                if self.conv_cur == self.last_conv or self.layer_cur > self.model_len-1:
                    self.layer_cur, self.conv_cur = self.init_layer, None        
                    epoch += 1
            elif self.type == 'resnet':
                self.layer_cur = self.getlayer(get_nextlayer(self.model, self.getidx(self.layer_cur)))
                if self.layer_cur == 0:
                    epoch += 1
        print('Pruned finish ! ...')  
              

class MulPruner(AutoPruner):
    def __init__(self, args, model, datasets, criterion, optimizer):
        #super(FBpruner, self).__init__(args, model, datasets, criterion, optimizer)
        self.model_len = len(model.features).s
        self.acc_d = 0
        self.acc_d_ = 0  
        self.a_base = 0.01
        self.b_base = 0.01
        self.alpha_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.beta_ = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.a1 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.a2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b1 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        self.b2 = torch.zeros(self.model_len, dtype = torch.float).to(device)
        #self.reset()
        super(MulPruner, self).__init__(args, model, datasets, criterion, optimizer)

    def reset(self):
        conv_idx = 0;
        for idx, layer in enumerate(self.model.features.children()):
            if isinstance(layer, nn.Conv2d):
                s0, s1, s2, s3 = layer.weight.size()
                self.param[idx], self.ch[idx], conv_idx = s0 * s1 * s2 *s3, s0, idx

            if isinstance(layer, nn.ReLU):
                self.ch[idx], self.ch[conv_idx] = self.ch[conv_idx], 0
                self.a1[idx], self.a2[idx] = self.a_base, 20
                self.b1[idx], self.b2[idx] = self.b_base, 20
        self.ch_, self.ch_log, self.ch_sqrt = self.ch, self.ch.log(), self.ch.sqrt()
        #self.ch_log = self.ch.log()
        #self.ch_sqrt = self.ch.sqrt()    

    '''def policy(self, method = 'mean'):
        self.alpha_[self.layer_cur] = self.alpha[self.layer_cur]
        self.beta_[self.layer_cur] = self.beta[self.layer_cur]
        #绛栫暐2
        if method == 'mean':
            self.alpha[self.layer_cur] += self.a1[self.layer_cur] * \
                self.ch[self.layer_cur].log() / (self.ch_log[self.layer_cur])            
        elif method == 'std':
            self.beta[self.layer_cur] += self.b1[self.layer_cur] * \
                self.ch[self.layer_cur].log() / (self.ch_log[self.layer_cur])            
        elif method == 'mstd':
            self.alpha[self.layer_cur] += self.a1[self.layer_cur] * \
                self.ch[self.layer_cur].log() / (self.ch_log[self.layer_cur])            
            self.beta[self.layer_cur] += self.b1[self.layer_cur] * \
                self.ch[self.layer_cur].log() / (self.ch_log[self.layer_cur])                        '''
    '''def get_goal(self, method = 'mean'):
        ## mean
        #print ('mean method - alpha', self.alpha[layer_cur])
        self.goal = 1- self.alpha[self.layer_cur] * (self.cal_mean + 0.0001).reciprocal()
        ## std
        #print ('std method - beta', self.beta[layer_cur])
        #self.goal = self.cal_std / self.beta[self.layer_cur] - 1
        ## std + mean
        #print('mean - ', self.alpha[layer_cur], ' and std - ', self.beta[layer_cur])
        #self.goal = self.cal_std / self.beta[self.layer_cur] - self.alpha[self.layer_cur] * (self.cal_mean + 0.0001).reciprocal()
    '''
    def get_iter(self, acc):
        iter1, iter2 = 3, 2
        iter1 = max(math.ceil((-10 * acc)), iter1)
        iter2 = max(math.ceil((-10 * acc) / 2), iter2)
        return iter1, iter2

    '''def channel_remove(self, policy='mean', layer_cur=0, iters=1, iters_=3, msg=True):
        if msg: print('Prune channel ...', end=' ')
        self.get_mean_std(layer_cur)    
        if msg: print('\rPC: get mean&std', end=' ')
        self.policy(method = policy)
        if msg: print('--> (s to a)', end=' ')
        self.get_goal()
        if msg: print('--> cal goal', end=' ')
        self.rchannel, filter_idx = self.goal.sort()
        if msg: print('--> sort importance')
        if policy == 'mean' or policy == 'mstd':
            print('\n[{0}]: alpha {1:.4f} --> {2:.4f} a: {3:.5f}'.format(
                self.layer_cur, self.alpha_[self.layer_cur], self.alpha[self.layer_cur], self.a1[self.layer_cur]))
        if policy == 'std' or policy == 'mstd':
            print('\n[{0}]: beta {1:.4f} --> {2:.4f} b: {3:.5f}'.format(
                self.layer_cur, self.beta_[self.layer_cur], self.beta[self.layer_cur], self.b1[self.layer_cur]))        
        #print('train the model ...', end='\r')

        if self.rchannel[0] < 0 and (self.rchannel<0).sum() < self.rchannel.size(0):
            channel =  filter_idx.masked_select(self.rchannel.le(0)) 
            model_ = self.model
            self.ch_[self.layer_cur] = (self.rchannel>0).sum()
            self.model = self.filter_remove(channel)

            torch.cuda.empty_cache()

            ##  add optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.prune_lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay) 

            print(self.model.features[self.conv_cur])
            if msg: print('weak Ch:', self.rchannel[:(self.rchannel<0).sum()], 'Total Ch:', self.rchannel.size(0))
            self.train_epochs(epochs=iters)
            return model_
        else: 
            self.train_epochs(epochs=iters_, msg=False)
            return self.model  '''  

    def single_layer_prune(self):

        print('prune single layer with facebace')
        self.train_epochs(epochs=5, msg=False)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        prune_loop = True
        err_count, weak, a = 0, 0, 1
        self.acc_d_ = self.acc_valid_top5 - acc_baseline
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            print('after prune {0:.4f}\n'.format( self.acc_valid_top5 - acc_baseline))

            #鏍规嵁绮惧害鏇存柊鍙傛暟
            acc_d_ = self.acc_valid_top5 - acc_baseline
            if acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                a = (1 + 0.5 * acc_d_) if acc_d_ > 0 else (1 + acc_d_)
                if acc_d_ < -0.2:   #杈冨樊鐨勬�鑳�                    weak += 1
                    if weak < 2:    #鍏佽�鐨勮�瀵熸湡锟�                        self.model = model
                        self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                        a /= 1 + weak
                    else:           #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                        weak, a = 0, 0
                else:   
                    weak , err_count = 0, 0

                #print ('a = {0:.4f}'.format(a))
                if a == 0:          #瓒呭嚭棰勬湡鐨勬儏鍐碉紝閲嶆柊瀹氫箟姝ラ暱
                    self.a1[self.layer_cur] = self.a_base
                    #print('绯绘暟', self.a1[self.layer_cur])
                    self.layer_cur = self.conv_cur
                    #prune_loop = False 
                    continue
                else:               #
                    self.a1[self.layer_cur] = a * self.a_base

                #淇濈暀濂界殑閫氶亾鏁伴噺
                self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                ##灏嗗綋鍓嶇殑绮惧害杩涜�澶囦唤
                #self.acc_d = self.acc_d_
                self.layer_cur = self.conv_cur

            else:
                err_count += 1
                if err_count == 3:
                    print(self.model)
                    self.model = torch.load('checkpoint/20/model').to(device)
                    self.layer_cur = self.layer_cur+1
                    err_count = 0
                    if self.conv_cur == self.last_conv:
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
        print('Pruned finish ! ...')

    def single_layer_fastprune(self):

        print('prune single layer with facebace')
        self.train_epochs(epochs=5, msg=False)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )
        prune_loop = True
        err_count, weak, a = 0, 0, 1
        acc_line = 0
        self.acc_d_ = self.acc_valid_top5 - acc_baseline
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)

            #鏍规嵁绮惧害鏇存柊鍙傛暟
            acc_d_ = self.acc_valid_top5 - acc_baseline
            acc_line = 0.9 * acc_line + 0.1 * acc_d_
            print('Prune_acc {0:.4f} baseline {1:.4f}\n'.format(acc_d_, acc_line))
            if acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                a = (1 + 0.5 * acc_d_) if acc_d_ > 0 else (1 + acc_d_)

                if (self.rchannel<0).sum() > 0:
                    if acc_d_ < acc_line:   #杈冨樊鐨勬�鑳�                        weak += 1
                        if weak < 2:    #鍏佽�鐨勮�瀵熸湡锟�                            self.model = model
                            self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                            a /= 1 + weak
                        else:           #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                            weak, a = 0, 0
                    else:   
                        weak , err_count = 0, 0

                    #print ('a = {0:.4f}'.format(a))
                    if a == 0:          #瓒呭嚭棰勬湡鐨勬儏鍐碉紝閲嶆柊瀹氫箟姝ラ暱
                        self.a1[self.layer_cur] = self.a_base
                        self.layer_cur = self.conv_cur
                        #prune_loop = False 
                        continue

                self.a1[self.layer_cur] = a * self.a_base

                #淇濈暀濂界殑閫氶亾鏁伴噺
                self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                ##灏嗗綋鍓嶇殑绮惧害杩涜�澶囦唤
                #self.acc_d = self.acc_d_
                self.layer_cur = self.conv_cur

            else:
                err_count += 1
                if err_count == 3:
                    print(self.model)
                    self.model = torch.load('checkpoint/20/model').to(device)
                    self.layer_cur = self.layer_cur+1
                    acc_line = 0
                    err_count = 0
                    if self.conv_cur == self.last_conv:
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
        print('Pruned finish ! ...')

    def whole_layer_prune(self):
        print('prune whole layer with facebace')
        self.train_epochs(epochs=5, msg=True)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )        
        prune_loop = True
        #iters = 0
        err_count = torch.zeros(self.model_len)
        weak = torch.zeros(self.model_len)
        a = 1
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)
            print('after prune {0:.4f}\n'.format( self.acc_valid_top5 - acc_baseline))

            #鏍规嵁绮惧害鏇存柊鍙傛暟
            acc_d_ = self.acc_valid_top5 - acc_baseline
            if acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                a = (1 + 0.5 * acc_d_) if acc_d_ > 0 else (1 + acc_d_)
                if acc_d_ < -0.2:   #杈冨樊鐨勬�鑳�                    weak[self.layer_cur] += 1
                    if weak[self.layer_cur] < 2:    #鍏佽�鐨勮�瀵熸湡锟�                        self.model = model
                        self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                        a /= 1 + weak[self.layer_cur]
                    else:           #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                        weak[self.layer_cur], a = 0, 0
                else:   
                    weak[self.layer_cur] , err_count[self.layer_cur] = 0, 0 

                if a == 0:          #瓒呭嚭棰勬湡鐨勬儏鍐碉紝閲嶆柊瀹氫箟姝ラ暱
                    self.a1[self.layer_cur] = self.a_base
                    #print('绯绘暟', self.a1[self.layer_cur])
                    self.layer_cur += 1
                    if self.conv_cur == self.last_conv:
                        self.layer_cur = 0                    
                    #prune_loop = False 
                    continue
                else:               #
                    self.a1[self.layer_cur] = a * self.a_base

                #淇濈暀濂界殑閫氶亾鏁伴噺
                self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0   
            else:
                err_count[self.layer_cur] += 1
                if err_count[self.layer_cur] == 3:
                    self.layer_cur = self.layer_cur+1
                    if self.conv_cur == self.last_conv:
                        print(self.model)
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0
        print('Pruned finish ! ...')

    def whole_layer_fastprune(self):
        print('prune whole layer with facebace')
        self.train_epochs(epochs=5, msg=True)
        acc_baseline = self.acc_valid_top5
        print('baseline: train {0:.3f} {1:.3f}  valid {2:.3f} {3:.3f}'.format(self.acc_train_top1, self.acc_train_top5, self.acc_valid_top1, self.acc_valid_top5) )        
        prune_loop = True
        #iters = 0
        err_count = torch.zeros(self.model_len)
        acc_line = (self.ch > 0).float() -1
        acc_line[self.model_len-2] = -1
        weak = torch.zeros(self.model_len)
        a = 1
        while prune_loop:
            iter1, iter2 = self.get_iter(self.acc_valid_top5 - acc_baseline)
            model = self.channel_remove(layer_cur=self.layer_cur, iters=iter1, iters_=iter2)

            #鏍规嵁绮惧害鏇存柊鍙傛暟
            acc_d_ = self.acc_valid_top5 - acc_baseline
            acc_line[self.layer_cur] = 0.9 * acc_line[self.layer_cur] + 0.1 * acc_d_
            print('Prune_acc {0:.4f} baseline {1:.4f}\n'.format(acc_d_, acc_line[self.layer_cur]))
            if acc_d_ > -1:         #鎬ц兘琛ㄧ幇杈冨ソ鐨勬儏锟�                a = (1 + 0.5 * acc_d_) if acc_d_ > 0 else (1 + acc_d_)

                if (self.rchannel<0).sum() > 0:
                    if acc_d_ < max(acc_line):   #杈冨樊鐨勬�鑳�                        weak[self.layer_cur] += 1
                        if weak[self.layer_cur] < 2:    #鍏佽�鐨勮�瀵熸湡锟�                            self.model = model
                            self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                            a /= 1 + weak[self.layer_cur]
                        else:           #瓒呭嚭瑙傛祴鏈熺殑鎯呭喌
                            weak[self.layer_cur], a = 0, 0
                    else:   
                        weak[self.layer_cur] , err_count[self.layer_cur] = 0, 0 

                    if a == 0:          #瓒呭嚭棰勬湡鐨勬儏鍐碉紝閲嶆柊瀹氫箟姝ラ暱
                        self.a1[self.layer_cur] = self.a_base
                        #print('绯绘暟', self.a1[self.layer_cur])
                        self.layer_cur += 1
                        if self.conv_cur == self.last_conv:
                            self.layer_cur = 0                    
                        #prune_loop = False 
                        continue

                self.a1[self.layer_cur] = a * self.a_base

                #淇濈暀濂界殑閫氶亾鏁伴噺
                self.ch[self.layer_cur] = self.ch_[self.layer_cur]
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0   
            else:
                err_count[self.layer_cur] += 1
                if err_count[self.layer_cur] == 3:
                    self.layer_cur = self.layer_cur+1
                    if self.conv_cur == self.last_conv:
                        print(self.model)
                        prune_loop = False     
                    continue
                self.model = model
                self.alpha[self.layer_cur] = self.alpha_[self.layer_cur]
                self.a1[self.layer_cur] *= 0.75  
                self.layer_cur += 1
                if self.conv_cur == self.last_conv:
                    self.layer_cur = 0
        print('Pruned finish ! ...')

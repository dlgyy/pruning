# import the library
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import *
from models import *
from torchvision import datasets, models, transforms

import time
import os
import copy
import argparse
import shutil

# global variable
global device
top5_flag = True


# optional parameter explanation
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='../data/VOC2012', metavar='DIR',
                    help='path to dataset')
#parser.add_argument('--arch', '-a', metavar='ARCH', default='darknet', #resnet18
                    #choices=model_names,
                    #help='model architecture: ' +
                        #' | '.join(model_names) +
                        #' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoceph number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,  #0.00001
                    metavar='LR', help='initial learning rate')#10-e5
parser.add_argument('--prune_lr', '--prune-learning-rate', default=0.00001, type=float,
                    metavar='prune LR', help='initial fine-tuning learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', #checkpoint/model_best.pth.tar
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--path', default='checkpoint/image/fvgg', metavar='DIR',
                    help='path to store')
parser.add_argument('--train', '-t',default='', action='store_true')
parser.add_argument('--valid', '-v',default='', action='store_true')
parser.add_argument('--prune', action = 'store_true')
parser.add_argument('--prune2', action = 'store_true')
parser.add_argument('--prune3', action = 'store_true')
parser.add_argument('--prune4', action = 'store_true')
parser.add_argument('--diff', default=0, type=float)


#class chvgg(nn.Module):
#    def __init__(self, features, num_classes=10):
#        super(chvgg, self).__init__()   
#        l = list(features.children())
#        self.features = torch.nn.Sequential(*l, nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)))
#        
#        
#    def forward(self, x):
#        x = self.features(x)
#        x = x.view(x.size(0), -1)
#        return x


## program entrance
def main():
    global args
    ## get optional parameter
    args = parser.parse_args()
    ## get dataset 
    #dataloaders = get_data()
    #print('load dataset finished ')
    
    # create model
    classes = 20   # the numb of class 
    ## read breakpoint
    #model = models.resnet18().to(device)
    model = models.resnet50(pretrained=True)#.to(device)
    #fc = model.fc
    #model.fc = nn.Linear(2048, 20)
    
    reslayer = [model.layer1, model.layer2, model.layer3, model.layer4]
    for l in range(4):
        #print(len(reslayer[l]))
        for i in range(len(reslayer[l])):
            if reslayer[l][i].downsample is None:
                size = reslayer[l][i].conv1.in_channels
                #print(reslayer[l][i].conv1.in_channels)
                reslayer[l][i].downsample = nn.Conv2d(size, size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                reslayer[l][i].downsample.weight.data = torch.eye(size).to(device).view([-1,size,1,1])
                reslayer[l][i].downsample.weight.data.requires_grad = False

    model = chresnet(model)
    #model = models.vgg16_bn(pretrained=True)
    #vgg = chvgg(model.features, num_classes=10)
    
    torch.save(model, 'resnet50')
    print(model)
    #print('创建时间: ', timestr)
            
if __name__=='__main__':
    main()
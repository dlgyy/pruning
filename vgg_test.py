# import the library
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import *
from models import *

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
parser.add_argument('--epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoceph number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  #0.00001
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
parser.add_argument('--path', default='checkpoint/cifar/', metavar='DIR',
                    help='path to store')
parser.add_argument('--train', '-t',default='', action='store_true')
parser.add_argument('--valid', '-v',default='', action='store_true')
parser.add_argument('--prune', action = 'store_true')
parser.add_argument('--prune2', action = 'store_true')
parser.add_argument('--prune3', action = 'store_true')
parser.add_argument('--prune4', action = 'store_true')
parser.add_argument('--diff', default=0, type=float)



## program entrance
def main():
    global args
    ## get optional parameter
    args = parser.parse_args()
    ## get dataset 
    
    
    dataloaders = get_cifar()#get_data()
    #print('load dataset finished ')
    
    ## create model
    classes = 10   # the numb of class 
    ### read breakpoint
    #model = torch.load('checkpoint/cifar/vgg16_bn').to(device)
    model = torch.load('checkpoint/cifar/model').to(device)
    ##model = torch.load('checkpoint/vgg16_bn').to(device)
    ##new_model = darknet(20).to(device)
    ##model = torch.load('checkpoint/image/fvgg/vgg16').to(device)
    ##model = models.vgg16_bn(pretrained=True)
    #print('load mode finished')
    print(model)
    print('创建时间: ', timestr)
    
    ### CPU -> GPU
    ##model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)            
    
    rlmodel = RLpruner2(args, model, dataloaders, criterion, optimizer)
    rlmodel.whole_layer_prune()
    
    
    
    if args.train:
        train_epochs(args, dataloaders, model, criterion, optimizer,args.epochs, store=True)
    #if args.valid:
        #validate(args, dataloaders['val'], model, criterion)
    #fp.close()
    #facc.close()
       
                
if __name__=='__main__':
    main()
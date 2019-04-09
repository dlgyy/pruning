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
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoceph number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,  #0.00001
                    metavar='LR', help='initial learning rate')#10-e5 0.0001
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
parser.add_argument('--path', default='./cr50', metavar='DIR',
                    help='path to store')
parser.add_argument('--train', '-t',default='', action='store_true')
parser.add_argument('--valid', '-v',default='', action='store_true')



## program entrance
def main():
    global args
    ## get optional parameter
    args = parser.parse_args()
    ## get dataset 
    dataloaders = get_cifar()#get_data()
    print('load dataset finished ')
    
    # create model
    classes = 20   # the numb of class 
    ## read breakpoint
    #model = cresnet50().to(device)
    model = torch.load('cr50model').to(device)
    #model = models.resnet50()
    #model = torch.load('model-0.01').to(device)
    #model = torch.load('model').to(device)
    #new_model = darknet(20).to(device)
    #model = torch.load('checkpoint/image/fvgg/vgg16').to(device)
    #model = models.vgg16_bn(pretrained=True)
    print('load mode finished')
    print(model)
    print('创建时间: ', timestr)
    
    ## CPU -> GPU
    #model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)            
    
    #automodel = AutoPruner(args, model, dataloaders, criterion, optimizer)
    #automodel.reset()
    #submodel = FBpruner(args, model, dataloaders, criterion, optimizer)
    #submodel.reset()
    #submodel.single_layer_prune()
    #submodel.single_layer_fastprune()
    #submodel.whole_layer_prune()
   #submodel.whole_layer_fastprune()
    #automodel.select_channel(layer_cur=0)
    #automodel.train_epochs(dataloaders, epochs = 1)
    #rlmodel = RLpruner(args, model, dataloaders, criterion, optimizer)
    
    #model = chresnet(model).to(device)
    #torch.save(model, 'resnet50')
    
    #m = vresnet50()
    
    #rlmodel = RLpruner2(args, model, dataloaders, criterion, optimizer)
    #rlmodel.whole_layer_prune()
    #print(rlmodel.getlayer([4,2,0]))
    
    #rlmodel.layer_cur = 0;
    #idx = [0, 0, 0]
    #for _ in range(60):
        #print(rlmodel.getidx(rlmodel.layer_cur), idx)
        ##block_remove(rlmodel.model, idx, torch.tensor([3,6,9]))
        ##idx = get_nextlayer(rlmodel.model, idx)  
        #rlmodel.resnet_remove(torch.tensor([3,6,9]))
        #idx = get_nextlayer(rlmodel.model, rlmodel.getidx(rlmodel.layer_cur))
        #rlmodel.layer_cur = rlmodel.getlayer(idx)
    #print(model)
    
    ##kd test
    #rlmodel = RLpruner(args, copy.deepcopy(new_model), dataloaders, criterion, optimizer)
    #rlmodel.train_kd_epochs(model, epochs=90)
    #rlmodel = RLpruner(args, copy.deepcopy(new_model), dataloaders, criterion, optimizer)
    #rlmodel.train_epochs(tmodel = model, epochs=90)
    #process
    if args.train:
        train_epochs(args, dataloaders, model, criterion, optimizer,args.epochs, pitch=100, store=True)
    if args.valid:
        validate(args, dataloaders['val'], model, criterion)
    fp.close()
    facc.close()

    

         
                
if __name__=='__main__':
    main()

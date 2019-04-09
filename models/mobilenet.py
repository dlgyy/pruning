import torch
import torch.nn as nn


'''class conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv_dw, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU(inplace=True)        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)        
        
        return out'''


class Mobilnet(nn.Module):
    def __init__(self, num_class=1000):
        super(Mobilnet, self).__init__()
        self.num_class = num_class
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
            

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),           
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.classifier = nn.Sequential(nn.Linear(1024, self.num_class))
        #self.fc = nn.Linear(1024, self.num_class)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

class Net(nn.Module):
    def __init__(self, num_class=1000):
        super(Net, self).__init__()
        self.num_class = num_class
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),            
            #conv_bn(  3,  32, 2), 
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(32, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            #conv_dw( 32,  64, 1),
            nn.Conv2d(64, 64, 3, 2, 1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            #conv_dw( 64, 128, 2),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),               
            #conv_dw(128, 128, 1),
            nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),               
            #conv_dw(128, 256, 2),
            nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(256, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),               
            #conv_dw(256, 256, 1),
            nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(256, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),               
            #conv_dw(256, 512, 2),
            nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),               
            #conv_dw(512, 512, 1),
            nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),              
            #conv_dw(512, 512, 1),
            nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),              
            #conv_dw(512, 512, 1),
            nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),              
            #conv_dw(512, 512, 1),
            nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),              
            #conv_dw(512, 512, 1),
            nn.Conv2d(512, 512, 3, 2, 1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),               
            #conv_dw(512, 1024, 2),
            nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(1024, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),               
            #conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.classifier = nn.Sequential(nn.Linear(1024, self.num_class))
        #self.fc = nn.Linear(1024, self.num_class)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x



def mobilenet(num_class=1000):
    model = Mobilnet(num_class)
    return model

def test():
    net = mobilenet()
    print(net)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y)
    
#test()
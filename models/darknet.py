import torch
import torch.nn as nn


class Darknet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(1024, self.num_classes, kernel_size=1, stride=1, padding=1),
            #nn.BatchNorm2d(num_classes), 
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5),
        )         
        '''self.classifier = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.AvgPool2d(kernel_size=3)
        )'''
        
    def forward(self, x):
        #x = self.features(x)
        x = self.features(x)
        #x = self.classifier(x)
        x = x.view(-1, self.num_classes)
        return x
        
'''class DARKNET(nn.Module):
    def __init__(self, features, num_classes = 1000):
        super(DARKNET, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, out_features, bias=True)
        )'''

class Darknet19(nn.Module):
    def __init__(self, num_classes = 1000):
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),              
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),              
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),              
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),      
            
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),                  
            
            nn.Conv2d(1024, self.num_classes, kernel_size=1, stride=1, padding=1),
            #nn.BatchNorm2d(num_classes), 
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5),
        )         
        '''self.classifier = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.AvgPool2d(kernel_size=3)
        )'''
        
    def forward(self, x):
        #x = self.features(x)
        x = self.features(x)
        #x = self.classifier(x)
        x = x.view(-1, self.num_classes)
        return x
        
def make_layers(cfg, batch_norm=False):    
    layers = []    
    in_channels = 3   
    for v in cfg:        
        if v == 'M':            
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]     
        elif v == 'A':
            #layers += [nn.AvgPool2d()]
            pass
        else:            
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)            
            if batch_norm:                
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]            
            else:                
                layers += [conv2d, nn.ReLU(inplace=True)]           
                in_channels = v    
    return nn.Sequential(*layers)
        
cfg = {    
    'A': [16, 'M', 32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M', 1024,  'P', 1024, 'A'],
    #'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def darknet(num_class=1000):
    model = Darknet(num_class)
    return model

def darknet19(num_class=1000):
    model = Darknet19(num_class)
    return model

def test():
    x = torch.randn(2, 3, 224, 224)
    net = Darknet(10)
    y = net(x)
    y
    
#test()    
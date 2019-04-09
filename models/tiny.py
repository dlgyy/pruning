import torch
import torch.nn as nn

class Tiny(nn.Module):
    def __init__(self, num_classes = 1000):
        super(Tiny, self).__init__()
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
            
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),               
            
            nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),                 
            
            nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),                     
            
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),           
            
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),                
            
            nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(num_classes), 
            nn.ReLU(inplace=True),
            
            #nn.MaxPool2d(kernel_size=2, stride=2),       
            nn.AvgPool2d(kernel_size = 14)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_classes)
        return x

    def tiny(num_class=1000):
        model = Tiny(num_class)
        return model
    
def test():
    x = torch.randn(2, 3, 224, 224)
    net = Tiny(10)
    y = net(x)
    y

#test()
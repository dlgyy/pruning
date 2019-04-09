import torch.nn as nn
import torch.utils.model_zoo as model_zoo


#__all__ = [
    #'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    #'vgg19_bn', 'vgg19',
#]


#model_urls = {
    #'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    #'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    #'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    #'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    #'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    #'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    #'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    #'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
#}


class VGGnet(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGGnet, self).__init__()
        self.features = features
        self.num_classes = num_classes
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    conv = nn.Conv2d(512, 10, kernel_size=1)
    layers += [conv]
    
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vggnet11(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['A']), **kwargs)


def vggnet11_bn(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['A'], batch_norm=True), **kwargs)


def vggnet13(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['B']), **kwargs)



def vggnet13_bn(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['B'], batch_norm=True), **kwargs)



def vggnet16(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['D']), **kwargs)


def vggnet16_bn(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['D'], batch_norm=True), **kwargs)


def vggnet19(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['E']), **kwargs)


def vggnet19_bn(pretrained=False, **kwargs):
    return VGGnet(make_layers_(cfg['E'], batch_norm=True), **kwargs)
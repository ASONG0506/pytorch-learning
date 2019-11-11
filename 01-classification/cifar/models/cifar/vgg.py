# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 
# usage:
# note: some samples are from :https://github.com/bearpaw/pytorch-classification
# --------------------

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm = False):
    layers=[]
    in_channels=3
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
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    model = VGG(cfg['A'], **kwargs)
    return model

def vgg11_bn(**kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm = True), **kwargs)
    return model

def vgg13(**kwargs):
    model = VGG(make_layers(cfg["B"]), **kwargs)
    return model

def vgg13_bn(**kwargs):
    model = VGG(make_layers(cfg["B"], batch_norm=True), **kwargs)
    return model

def vgg16(**kwargs):
    model = VGG(make_layers(cfg["D"]),  **kwargs)
    return model

def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg["D"], batch_norm = True), **kwargs )
    return model

def vgg19(**kwargs):
    model = VGG(make_layers(cfg["E"]), **kwargs)
    return model

def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg["E"], batch_norm=True), **kwargs)
    return model




if __name__ == "__main__":
    print ("test the vgg model series")

    # test the realization of the vgg net using simple dataset
    net = vgg16()

    # input data
    input_data = torch.randn(2,3,32,32)

    # do inference
    output = net(input_data)
    print(output.size())
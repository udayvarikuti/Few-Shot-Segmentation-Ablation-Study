"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.features = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        self.features.aux_classifier=Identity()
        self.features
        #print(type(self.features))

    def forward(self, x):
        y=self.features(x)['out']
        #print(type(y))
        #print(y)
        return y


# a=Encoder()
# print(a.features)
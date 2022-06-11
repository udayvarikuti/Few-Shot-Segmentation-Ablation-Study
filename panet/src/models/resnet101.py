"""
Encoder for few shot segmentation (Resnet101)
"""

import torch
import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.features = torchvision.models.resnet101(pretrained=True) 
        self.features.avgpool=Identity()
        self.features.fc=nn.Conv2d(
                                    in_channels=2048,
                                    out_channels=512,
                                    kernel_size=1,
                                    stride=1
                                    )
        #print(self.features)

    def forward(self, x):
        y=self.features(x)
        #print(y.shape)
        return y


# a=Encoder()
# print(Encoder)
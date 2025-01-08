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
        self.features = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.features.aux_classifier=Identity()
        self.features.classifier[4]= nn.Conv2d(
                                                in_channels=256,
                                                out_channels=512,
                                                kernel_size=1,
                                                stride=1
                                                )
        #self.features.classifier.requires_grad=False
        #print(self.features)

    def forward(self, x):
        y=self.features(x)['out']
        #print(y.shape)
        return y


a=Encoder()

# print(Encoder)
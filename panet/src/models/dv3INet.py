"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.features = smp.DeepLabV3(encoder_name='resnet50', encoder_weights='imagenet')
        self.features.segmentation_head= nn.Conv2d(
                                                in_channels=256,
                                                out_channels=512,
                                                kernel_size=1,
                                                stride=1
                                                )
        #self.features.encoder.requires_grad=False
        #print(self.features)

    def forward(self, x):
        y=self.features(x)
        #print(y.shape)
        return y


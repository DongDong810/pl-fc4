# Reference - fc4.py / fcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Union
from models.SqueezeNet_v1 import SqueezeNet

class FC4(nn.Module):
    def __init__(self, cfg):
        super(FC4, self).__init__()
        self.cfg = cfg

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        squeeze = SqueezeNet()
        self.backbone = nn.Sequential(*list(squeeze.children())[0][:12])
  
        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 4 if self.cfg.use_confidence_weighted_pooling else 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB color for the illuminant of the input image
        @param x: the image for which the color of the illuminant has to be estimated
        @return: the color estimate as a Tensor. If confidence-weighted pooling is used, the per-path color estimates and the confidence weights are returned as well (used for visualizations)
        """

        ret_dict = {}

        x = x.float()
        x = self.backbone(x)
        out = self.final_convs(x)

        # Confidence-weighted pooling: "out" is a set of semi-dense feature maps
        if self.cfg.use_confidence_weighted_pooling:
            # Per-patch color estimates (first 3 dimensions)
            rgb = F.normalize(out[:, :3, :, :], dim=1)

            # Confidence (last dimension)
            confidence = out[:, 3:4, :, :]

            # Confidence-weighted pooling
            pred = F.normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

            ret_dict['y_hat'] = pred
            ret_dict['rgb'] = rgb
            ret_dict['confidence'] = confidence

            return ret_dict

        # Summation pooling
        pred = F.normalize(torch.sum(torch.sum(out, 2), 2), dim=1)

        ret_dict['y_hat'] = pred

        return ret_dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# Fire module
class Fire(nn.Module):
  def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
    super(Fire, self).__init__()
    self.in_channels = in_channels
    self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1)
    self.expand1x1 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand1x1_channels, kernel_size=1)
    self.expand3x3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand3x3_channels, kernel_size=3, padding=1)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.squeeze(x))
    return torch.cat([self.relu(self.expand1x1(x)), self.relu(self.expand3x3(x))], dim=1) # expand1x1, expand3x3 -> relu -> concatenate


# Backbone Squeeze Net (version 1.1)
class SqueezeNet(nn.Module):
  def __init__(self, num_classes=1000):
    super(SqueezeNet, self).__init__()
    self.num_classes = num_classes
    self.features = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0),    # conv1
      nn.ReLU(inplace=True),                                                            # relu
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),                            # pool / ceil_mode=True : round up the size
      Fire(64, 16, 64, 64),                                                             # fire2
      Fire(128, 16, 64, 64),                                                            # fire3
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),                            # pool (padding=same?)
      Fire(128, 32, 128, 128),                                                          # fire4
      Fire(256, 32, 128, 128),                                                          # fire5
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),                            # pool (padding=same?)
      Fire(256, 48, 192, 192),                                                          # fire6
      Fire(384, 48, 192, 192),                                                          # fire7
      Fire(384, 64, 256, 256),                                                          # fire8
      Fire(512, 64, 256, 256),                                                          # fire9
    )

    final_conv = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)    # conv10
    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5),                                                               
      final_conv,                                                                     
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d((1, 1))                                                      # global average pooling (B, C, H, W) -> (B, C, 1, 1)                                                         
    )
    
    for m in self.modules(): # checking all layers
      if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x.view(x.size(0), self.num_classes) # change size to (B, 1000)

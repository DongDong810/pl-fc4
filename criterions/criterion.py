import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from torch import Tensor

class MasterCriterion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.key_pairs = cfg.criterion.key_pairs # {choose loss}
        
        self.mod_dict = {}
        self.mod_dict['l1_loss'] = L1_loss(cfg)
        self.mod_dict['cross_entropy'] = Cross_entropy(cfg)
        self.mod_dict['angular_loss'] = Angular_loss(cfg)
        #################### add more losses ####################

    # Compute specfic losses -> weighted sum for total loss
    def forward(self, pred_dict, gt_dict, phase):
        
        loss_dict = {}
        total_loss = 0
        
        # Total loss = weighted sum of losses
        for loss_key in self.key_pairs:
            mod_key = self.cfg.criterion[loss_key].mod  # which loss?
            alpha = self.cfg.criterion[loss_key].alpha  # weight
            
            loss = self.mod_dict[mod_key](pred_dict, gt_dict) # compute specific loss
            loss_dict[f'{phase}-{loss_key}'] = loss
            total_loss += (alpha * loss)
        
        loss_dict[f'{phase}-total_loss'] = total_loss

        return loss_dict

########################### LOSS FUNCTIONS ###########################

class L1_loss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.L1 = nn.L1Loss()

    def forward(self, pred_dict, gt_dict):
        loss = self.L1(pred_dict['y_hat'], gt_dict['y'])

        return loss
    

class Cross_entropy(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred_dict, gt_dict):
        loss = self.cross_entropy(pred_dict['y_hat'], gt_dict['y'])

        return loss

class Angular_loss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def _compute_angular_loss(self, pred, label, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(F.normalize(pred, dim=1) * F.normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)

    def forward(self, pred_dict, gt_dict):
        loss = self._compute_angular_loss(pred_dict['y_hat'], gt_dict['y'])

        return loss
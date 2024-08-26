import os, sys
import scipy.io 
import torch 
import numpy as np
from datasets.augment import DataAugmenter

sys.path.insert(1, os.path.abspath('..'))
from typing import Tuple
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from auxiliary.utils import get_mcc_coord, load_image, load_image_without_mcc, normalize, bgr_to_rgb, correct, linear_to_nonlinear, hwc_to_chw


class ColorCheckerDataset(Dataset):
    def __init__(self, cfg, phase, fold_nums):
        super(ColorCheckerDataset, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.root_dir = cfg.data.root
        self.data = self._load_metadata(fold_nums)
        self.augmenter = DataAugmenter(cfg)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        """
        @return fn: file name
        @return illum : GT illuminant
        @return img : image without color checker (for training)
        """
        batch_dict = {}

        _, fn, r, g, b = self.data[idx].strip().split(' ')
        img = load_image_without_mcc(fn, get_mcc_coord(fn))
        illum = [float(r), float(g), float(b)]
        
        # Modify data
        if self.phase == 'train':
            img, illum = self.augmenter.augment(img, illum)
        else:
            img = self.augmenter.crop(img)
        
        img = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(img))))

        img = torch.from_numpy(img.copy())
        illum = torch.from_numpy(np.array(illum.copy(), dtype=np.float32))
        
        # Store in batch_dict (x: input image, y: GT illuminant label, fn: file name)
        batch_dict['x'] = img
        batch_dict['y'] = illum
        batch_dict['fn'] = fn

        return batch_dict

    # Metadata only with the fold
    def _load_metadata(self, fold_nums):

        folds_fn = os.path.join(self.root_dir, 'folds.mat')
        folds = scipy.io.loadmat(folds_fn)

        meta = []

        for f in fold_nums:
            img_idx = folds["tr_split" if self.phase == "train" else "te_split"][0][f][0] # Indices of image folds
            meta_path = '../data/gehler/metadata.txt'
            metadata = open(meta_path, 'r').readlines()
            for i in img_idx:
                meta.append(metadata[i - 1])

        return meta


def get_loader(cfg, phase):
    """
    @param cfg: config file
    @param pahse : ['train', 'valid', 'test']
    @param fold_nums : list  e.g. [0, 1, 2] - train with 0, 1 & valid with 2
    """
    if phase == 'train':
        dataset = ColorCheckerDataset(cfg, phase, cfg.k_folds.train) # [0, 1]
    elif phase == 'valid':
        dataset = ColorCheckerDataset(cfg, phase, cfg.k_folds.val) # [2]
    else:
         dataset = ColorCheckerDataset(cfg, phase, cfg.k_folds.test) # [0]
    
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True, # for GPU
    )
    return loader

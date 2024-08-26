# Import for dataset loader
from datasets.colorchecker import get_loader as get_colorchecker_loader

def get_loader(cfg, phase):
    if cfg.data.name == 'colorchecker':
        return get_colorchecker_loader(cfg, phase)
    else:
        raise NotImplementedError
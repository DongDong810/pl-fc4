################ Modification ################
import sys,os,random,torch,importlib,time
import torch.nn as nn
import wandb
import torch.distributed as dist
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Insert path for finding module (order: . -> ../../ -> ..)
sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('../../'))
from utils.setup import init_path_and_expname,get_callbacks,get_logger,get_trainer_args
from datasets.unified_loader import get_loader
from criterions.criterion import MasterCriterion
from torchsummary import summary
import pytorch_lightning as pl

# Main - e.g. python main.py model.name=fc4 model.ver=v1 model.solver=v1
if __name__ == '__main__':
    # Import default config file
    cfg = OmegaConf.load(f'../configs/default.yaml')
    # Read from command line (model.name / model.ver / model.solver)
    cfg_cmd = OmegaConf.from_cli()  # return dictionary object
    """
    {
        "model": {
            "name": "FC4",
            "ver": "v1",
            "solver": "v1"
        }
    }
    """
    # Merge model specific config file
    if "model" in cfg_cmd  and 'name' in cfg_cmd.model: # check key (model.name)
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f'../configs/{cfg_cmd.model.name}.yaml'))  # model from command line
    else:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f'../configs/{cfg.model.name}.yaml'))  # model from default.yaml
    # Merge cfg from command line
    cfg = OmegaConf.merge(cfg, cfg_cmd)  # default.yaml + model.yaml + command line

    # Path and exp_name configuration
    init_path_and_expname(cfg)  # only done in master process (setup.py)
    pl.seed_everything(cfg.seed)

    # Dataloader
    dataloader = {
        'train': get_loader(cfg, 'train') if cfg.mode == 'train' else None,
        'valid': get_loader(cfg, 'valid') if cfg.mode == 'train' else None,
        'test' : get_loader(cfg, 'test') if cfg.mode == 'test' else None
    }

    # Dynamic Model module import
    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')  # import models/FC4_v1.py
    network_class = getattr(network_mod, cfg.model.name)  # get class named "cfg.model.name"
    network = network_class(cfg)  # make instance with the class

    # Loss
    loss = MasterCriterion(cfg)  # make instance from criterions/criterion.py

    # Dynamic Solver module import
    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')  # import solvers.fc4_v1.py
    solver_class = getattr(solver_mod, 'Solver')  # get class named "Solver"
    solver = solver_class(net=network,
                          loss=loss,
                          **(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))  # make instance with the class

    # Summary of model
    # summary(network, (1, 28, 28)) # (in_channels, image height, image width)

    # Init trainer
    trainer_args = get_trainer_args(cfg)  # return dictionary with train args
    trainer = pl.Trainer(**trainer_args)  # make trainer

    # Automatically train/test
    if cfg.mode == 'train':
        trainer.fit(
            model=solver,
            train_dataloaders=dataloader['train'],
            val_dataloaders=dataloader['valid'],
            ckpt_path=cfg.load.ckpt_path if cfg.load.load_state else None
        )
    
    elif cfg.mode == 'test':
        trainer.test(
            model=solver,
            dataloaders=dataloader['test']
        )


from gencls.models.model_cls import ClsModel
import torch.optim as optim_module
from  torch.optim import lr_scheduler
import torch.nn as nn 
import copy

def build_model(config):
    model = ClsModel(
        num_classes=config['Model']['num_classes'],
        backbone_config=config['Model']
    )
    return model

def build_optimizer(config, model):
    config = copy.deepcopy(config)
    optimizer_config = config['Optimizer']
    optim_name =  optimizer_config.pop('name')
    print(optim_name)
    lr = optimizer_config.pop('lr')
    optimizer =  getattr(optim_module, optim_name)(
        params=model.parameters(),
        lr=lr,
        **optimizer_config
    )
    return optimizer


def build_scheduler(config, optimizer, step_per_epoch=None):
    config = copy.deepcopy(config)
    scheduler_config = config['Scheduler']
    scheduler_name =  scheduler_config.pop('name')
    if scheduler_name == 'OneCycleLR': 
        scheduler_config['epochs'] = config['Common']['epochs']
        scheduler_config['steps_per_epoch'] = step_per_epoch

    scheduler = getattr(lr_scheduler, scheduler_name)(
        optimizer,
        **scheduler_config
    )
    return scheduler

def build_loss(config):
    config = copy.deepcopy(config)
    loss_config = config['Loss']
    loss_name =  loss_config.pop('name')
    
    loss =  getattr(nn, loss_name)()
    return loss
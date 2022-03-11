import torch 
from tools.misc.logger import get_logger
logger = get_logger()

def save_checkpoint(checkpoint_path, 
                    model, 
                    epoch, 
                    best_metric, 
                    optimizer=None, 
                    scheduler=None):

    torch.save({
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric

    }, checkpoint_path)

def load_checkpoint(checkpoint_path, 
                    model, 
                    optimizer=None, 
                    scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    return model, optimizer, scheduler, epoch, best_metric
    

def save_weight(model, weight_path):
    torch.save(model.state_dict(), weight_path)

def load_weight(weight_path, model):
    checkpoint = torch.load(weight_path)
    for name, param in model.named_parameters():
        if name not in checkpoint:
            logger.info('{} not found'.format(name))
        elif checkpoint[name].shape != param.shape:
            logger.info('{} missmatching shape, required {} but found {}'.format(
                name, param.shape, checkpoint[name].shape))
            del checkpoint[name]
    model.load_state_dict(checkpoint, strict=False)
    return model

def move_optimize_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

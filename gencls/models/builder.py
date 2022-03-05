from gencls.models.model_cls import ClsModel

def build_model(config):
    model = ClsModel(
        num_classes=config['Model']['num_classes'],
        backbone_config=config['Model']
    )
    return model

def build_optimizer(config):
    pass 

def build_scheduler(config):
    pass
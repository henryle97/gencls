import os.path as osp
import os
import torch
import tqdm
import time

from gencls.dataset.builder import build_dataloader
from gencls.engine.evaluation import evaluate, test
from gencls.engine.train import train_one_epoch
from gencls.evaluation import build_metric
from gencls.models.builder import build_loss, build_model, build_optimizer, build_scheduler
from gencls.dataset.postprocess import build_postprocess
from tools.misc.logger import get_logger
from tools.misc.save_load_model import save_checkpoint, load_checkpoint, save_weight, load_weight, move_optimize_to_device
from tools.misc.average_meter import AverageMeter
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.benchmark = True

class Engine:
    def __init__(self, config, mode='train'):
        """ Engine class

        Args:
            config (dict): config dict 
            mode (str, require): train | eval | infer | pseudo. Defaults to 'train'.
        """
        assert mode in ['train', 'eval', 'infer', 'pseudo']

        self.config = config
        self.mode = mode
        self.use_gpu = config['Common']['use_gpu']
        self.exp_dir = config['Common']['exp_dir']

        # create exp dir 
        if not osp.exists(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)
        
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        # logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.exp_dir, f'{timestamp}.log')
        self.logger = get_logger(log_file=log_file, log_level='INFO')
        self._init_info_dict()
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.exp_dir)
        
        self.logger.info(config)

        # build dataloader
        if mode == 'train':
            self.train_dataloader = build_dataloader(config, mode='train')
            # from IPython import embed; embed()
            self.logger.info(f"Num of batch of train loader: {len(self.train_dataloader)}")
            
        if mode == 'eval' or (mode == 'train' and config['Common']['eval_during_training']):
            self.val_dataloader = build_dataloader(config, mode='eval')
            self.logger.info(f"Num of batch of val loader: {len(self.val_dataloader)}")
            self.eval_metric_func = build_metric(config)
        if mode in ['infer', 'pseudo']:
            self.postprocess_func = build_postprocess(
                self.config['Infer']['PostProcess'])
            self.infer_dataloader = build_dataloader(config, mode)
            self.logger.info(f"Num of batch of infer loader: {len(self.infer_dataloader)}")

        # build model 
        self.model = build_model(config)

        # build opt + scheduler + loss
        if mode in ['train']:
            self.optimizer = build_optimizer(config, self.model)
            self.scheduler = build_scheduler(config, self.optimizer, step_per_epoch=len(self.train_dataloader))
            self.criterion = build_loss(config)

        # load model
        if config['Common']['resume_from']:
            resume_path = config['Common']['resume_from']
            self.model, self.optimizer, self.scheduler, \
            self.epoch, self.best_metric = load_checkpoint(
                checkpoint_path=resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            move_optimize_to_device(self.optimizer, self.device)
            self.logger.info(f"Resume from: {resume_path}")
        elif config['Common']['pretrained_model']:
            pretrained_path = config['Common']['pretrained_model']
            load_weight(pretrained_path, self.model)
            self.logger.info(f"Load pretrained weight from: {pretrained_path}")
        
        self.model = self.model.to(self.device)
        
        # amp 
        self.grad_scaler = None
        if self.config['Common']['amp']:
            cuda = self.device != 'cpu'
            self.logger.info("Training with AMP mode")
            self.grad_scaler = amp.GradScaler(enabled=cuda)

        # save config
        config.save(osp.join(self.exp_dir, "config.yml"))
        

    def train(self):
        assert self.mode == 'train'
        tic = time.time()
        best_metric = {
            'metric': 0.0,
            'epoch': 0
        }
        best_weight_path = osp.join(self.exp_dir, "best.pt")
        last_weight_path = osp.join(self.exp_dir, "last.pt")
        self.model.train()
        for epoch_id in range(1, self.config['Common']['epochs']+1):
            train_one_epoch(engine=self, 
                            epoch_id=epoch_id,
                            print_batch_step=self.config['Common']['print_per_step']
                            )
            if self.config['Common']['eval_during_training'] \
                    and epoch_id % self.config['Common']['eval_per_epoch'] == 0:
                acc = self.eval(epoch_id)
                if acc > best_metric['metric']:
                    best_metric['metric'] = acc 
                    best_metric['epoch'] = epoch_id
                    save_weight(self.model, best_weight_path)
            save_checkpoint(
                            checkpoint_path=last_weight_path,
                            model=self.model,
                            epoch=epoch_id,
                            best_metric=best_metric,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler
            )
            self.logger.info(f"[Eval][Epoch {epoch_id}/{self.config['Common']['epochs']}]: Best Acc {best_metric['metric']} at epoch {best_metric['epoch']}")
        self.logger.info("Total time training: {:.2f}".format(time.time() - tic))

    def eval(self, epoch_id):
        self.model.eval()
        eval_result = evaluate(self, epoch_id)
        self.model.train()
        return eval_result

    def test(self):
        self.model.eval()
        metric_result = test(self)
        return metric_result


    def infer_on_dataloader(self):
        self.model.eval()
        results = []
        # from IPython import embed; embed()
        for idx, batch_data in tqdm.tqdm(enumerate(self.infer_dataloader)):
            
            batch_data = self.transfer_batch_to_device(batch_data)
            outputs = self.model(batch_data['img'])
            result_batch = self.postprocess_func(outputs,
                                                 filenames=batch_data['img_path'],
                                                 multiple_label=False)
            results.extend(result_batch)
            if idx == 5:
                break
        return results

    def transfer_batch_to_device(self, batch):
        
        for key in batch.keys():
            if key == 'img_path':
                continue
            batch[key] = batch[key].to(self.device, non_blocking=True)
        return batch

    def _init_info_dict(self):
        self.time_info = {
            'data_time': AverageMeter(
                'data_time', fmt='.2f', postfix=" s,"
            ),
            'forward_time': AverageMeter(
                'forward_time', fmt='.2f', postfix=" s,"
            ),
            'backward_time': AverageMeter(
                'forward_time', fmt='.2f', postfix=" s,"
            ),
        }

        self.loss_info = AverageMeter(
            'loss_info', '.4f'
        )
        self.metric_info = AverageMeter(
            'metric_info', '.4f'
        )

    

import os.path as osp
import os
from cv2 import log

import torch
from torch.utils.data import DataLoader
import tqdm

from gencls.dataset.simple_dataset import SimpleDataset
from gencls.models.builder import build_model, build_optimizer, build_scheduler
from gencls.dataset.postprocess import build_postprocess
from tools.misc.get_image_list import get_image_list
from tools.misc.logger import get_root_logger
from tools.misc.save_load_model import save_checkpoint, load_checkpoint, save_weight, load_weight, move_optimize_to_device
import time

class Engine:
    def __init__(self, config, mode='train'):
        """ Engine class

        Args:
            config (dict): config dict 
            mode (str, require): train | test | infer | pseudo. Defaults to 'train'.
        """
        self.config = config
        self.mode = mode
        self.use_gpu = config['Common']['use_gpu']
        self.exp_dir = config['Common']['exp_dir']
        if not osp.exists(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = log_file = osp.join(self.exp_dir, f'{timestamp}.log')
        self.logger = get_root_logger(log_file=log_file, log_level='INFO')

        # Model + optimizer + scheduler 
        self.model = build_model(config=config)
        self.optimizer = build_optimizer(config)
        self.scheduler = build_scheduler(config)

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
        if mode in ['infer', 'pseudo']:
            self.postprocess_func = build_postprocess(
                self.config['Infer']['PostProcess'])
            if mode == 'pseudo':
                image_paths = get_image_list(self.config['Infer']['img'])

            elif mode == 'infer':
                if osp.isfile(self.config['Infer']['img']):
                    image_paths = [self.config['Infer']['img']]
                else:
                    image_paths = get_image_list(self.config['Infer']['img'])

            infer_dataset = SimpleDataset(
                image_paths=image_paths,
                transform_ops=self.config['Infer']['transforms']
            )
            self.logger.info(f"Number of images in infer_dataset: {infer_dataset.__len__()}")
            self.infer_dataloader = DataLoader(
                dataset=infer_dataset,
                batch_size=self.config['Infer']['batch_size'],
                num_workers=self.config['DataLoader']['num_workers'],
                pin_memory=self.config['DataLoader']['pin_memory'],
                shuffle=self.config['Infer']['shuffle']
            )

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
            # if idx == 5:
            #     break
        return results

    def transfer_batch_to_device(self, batch):
        for key in batch.keys():
            if key == 'img_path':
                continue
            batch[key] = batch[key].to(self.device)
        return batch

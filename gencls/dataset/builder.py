from tools.misc.get_image_list import get_image_list
from .lmdb_dataset import LmdbDataset
from .simple_dataset import SimpleDataset
import os.path as osp
from torch.utils.data import DataLoader


def build_dataset(config, mode='train'):
    assert mode in ['train', 'eval', 'infer', 'pseudo'], 'invalid mode engine'

    if config['Dataset']['type'] == 'SimpleDataset':
        if mode in ['train', 'eval']:
            
            if mode == 'train':
                label_path = config['Dataset']['train_label_path']
            else:
                label_path = config['Dataset']['val_label_path']
            assert label_path is not None, 'must set label path'
            mode_key = mode.title()
            dataset = SimpleDataset(
                image_root=config['Dataset']['image_root'],
                label_path=label_path,
                transform_ops=config['Dataset'][mode_key]['transforms']
            )
        else:
            if mode == 'infer':
                if osp.isfile(config['Infer']['img']):
                    image_paths = [config['Infer']['img']]
                else:
                    image_paths = get_image_list(config['Infer']['img'])
            elif mode == 'pseudo':
                image_paths = get_image_list(config['Infer']['img'])

            dataset = SimpleDataset(
                image_root=None,
                image_paths=image_paths,
                transform_ops=config['Infer']['transforms']
            )
    elif config['Dataset']['type'] == 'LmdbDataset':
        if mode in ['train', 'eval']:
            
            if mode == 'train':
                label_path = config['Dataset']['train_label_path']
            else:
                label_path = config['Dataset']['val_label_path']
                
            assert label_path is not None, 'must set label path'
            mode_key = mode.title()
            dataset = LmdbDataset(
                label_path=label_path,
                transform_ops=config['Dataset'][mode_key]['transforms']
            )
    return dataset


def build_dataloader(config, mode='train'):
    dataset = build_dataset(config, mode=mode)
    num_workers, pin_memory = config['DataLoader']['num_workers'], \
                            config['DataLoader']['pin_memory']

    key_dict = mode.title() if mode != 'pseudo' else 'Infer'
    if mode == 'pseudo' or mode == 'infer':
        batch_size, shuffle = config[key_dict]['batch_size'],  \
                            config[key_dict]['shuffle'] 
    else:
        batch_size, shuffle = config['Dataset'][key_dict]['batch_size'],  \
                            config['Dataset'][key_dict]['shuffle'] 
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )
    return dataloader

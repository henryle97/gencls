
from http.cookiejar import LoadError
import cv2
import os.path as osp
import random
import torch
import tqdm 
from torch.utils.data import Dataset 
from gencls.dataset.preprocess.create_operators import create_operators
from gencls.dataset.preprocess.transform import transform
import random 

class BaseDataset(Dataset):
    def __init__(self, 
                transform_ops=None,
                ):
        if transform_ops:
            self.transform_ops = create_operators(transform_ops)
        self.image_paths = []
        self.labels = []

    def _get_data(self, idx):
        """

        Return:
            img, label, img_path
        """
        pass

    def _load_annos(self):
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img, label, img_path = self._get_data(idx)
            
            
            img = img.transpose((2, 0, 1)) # HxWxC --> CxHxW

        except Exception as err:
            print(err)
            rnd_idx = random.randint(0, self.__len__() - 1)
            self.__getitem__(rnd_idx)
        sample = {
            'img': torch.FloatTensor(img),
            'label': float(label),
            'img_path': img_path
        }
        return sample

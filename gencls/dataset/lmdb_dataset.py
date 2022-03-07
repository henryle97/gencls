
from http.cookiejar import LoadError
import cv2
import os.path as osp
import random
import numpy as np
import tqdm
import os.path as osp

from gencls.dataset.base_dataset import BaseDataset
from gencls.dataset.preprocess.create_operators import create_operators
from gencls.dataset.preprocess.transform import transform

import lmdb 
import six
from PIL import Image
class LmdbDataset(BaseDataset):
    def __init__(self,
                 image_root=None,
                 label_path=None,
                 transform_ops=None,
                 ):
        super(LmdbDataset, self).__init__()
        self.image_root = image_root
        self.label_path = label_path
        if transform_ops:
            self.transform_ops = create_operators(transform_ops)
        self.env = lmdb.open(
            label_path,
            max_readers=8,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.txn = self.env.begin(write=False)
        nSamples = int(self.txn.get('num-samples'.encode()))
        self.nSamples = nSamples

    def read_buffer(self, idx):
        img_file = 'image-%09d'%idx
        label_file = 'label-%09d'%idx
        path_file = 'path-%09d'%idx
        
        imgbuf = self.txn.get(img_file.encode())
        
        label = self.txn.get(label_file.encode()).decode()
        img_path = self.txn.get(path_file.encode()).decode()

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
    
        return buf, label, img_path
    
    def _get_data(self, idx):
        buf, label, img_path = self.read_buffer(idx)
        img = Image.open(buf).convert('RGB')  
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
        if self.transform_ops:
            img = transform(img, self.transform_ops)

        return img, label, img_path

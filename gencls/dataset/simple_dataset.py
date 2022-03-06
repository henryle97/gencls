
from http.cookiejar import LoadError
import cv2
import os.path as osp
import random
import tqdm
import os.path as osp

from gencls.dataset.base_dataset import BaseDataset
from gencls.dataset.preprocess.create_operators import create_operators
from gencls.dataset.preprocess.transform import transform


class SimpleDataset(BaseDataset):
    def __init__(self,
                 image_root=None,
                 label_path=None,
                 image_paths=None,
                 transform_ops=None,
                 ):
        super(SimpleDataset, self).__init__()
        self.image_root = image_root
        self.label_path = label_path
        self.pseudo_image_paths = image_paths
        if transform_ops:
            self.transform_ops = create_operators(transform_ops)
        self.image_paths = []
        self.labels = []
        self._load_annos()
        self.nSamples = len(self.image_paths)

    def _load_annos(self):
        '''
        line format: img_path\tlabel
        '''
        if self.pseudo_image_paths is not None:
            self.image_paths = self.pseudo_image_paths
            self.labels = [0] * len(self.image_paths)

        elif osp.exists(self.label_path):
            with open(self.label_path, 'r', encoding='utf8') as f:
                for line in tqdm.tqdm(f):
                    item = line.strip().split("\t")
                    if len(item) != 2:
                        raise LoadError("Label format error")
                    img_path, label = item
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))
        else:
            raise NotImplementedError("Cannot load annotation!")


    def _get_data(self, idx):
        print("IDX ", idx)
        if self.image_root:
            img_path = osp.join(self.image_root, self.image_paths[idx])
        else:
            img_path = self.image_paths[idx]

        img = cv2.imread(img_path)
        # if img is None: 
        #     print("Get image None")
        #     raise Exception
        if self.transform_ops:
            img = transform(img, self.transform_ops)

        label = self.labels[idx]
        return img, label, img_path

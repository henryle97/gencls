import math
import numpy as np 
import torch

from gencls.dataset.postprocess import build_postprocess
from gencls.dataset.preprocess.create_operators import create_operators
from gencls.dataset.preprocess.transform import transform
from gencls.models.builder import build_model


class Predictor:
    def __init__(self, config, use_gpu=False):
        self.config = config 
        self.device = torch.device("cuda") if use_gpu else torch.device('cpu')
        self.model = build_model(config)
        self.transform_ops = create_operators(config['Infer']['transforms'])
        self.postprocess_func = build_postprocess(self.config['Infer']['PostProcess'])
        self.batch_size = self.config['Infer']['batch_size']

    def infer(self, images):
        '''
        params:
            images (list): list images

        returns:
            (list[dict]): keys: {'class_ids', 'scores'}

        '''
        image_batches = self.split_batch(images, batch_size=self.batch_size)
        pre_batches = []
        for batch in image_batches:
            pre_batch = []
            for image in batch:
                pre_image = transform(image, ops=self.transform_ops)
                pre_batch.append(pre_image)
            pre_batch = np.asarray(pre_batch)
            pre_batch = torch.FloatTensor(pre_batch)
            pre_batches.append(pre_batch)

        outputs = []
        for batch in pre_batches:
            batch = batch.to(self.device)
            output = self.model(batch)
            output = self.postprocess_func(output)
            outputs.extend(output)
        return outputs

    def split_batch(self, list_images, batch_size):
        image_batches = []
        total_images = len(list_images)
        num_batches = math.ceil(total_images / batch_size)
        for batch_id in range(num_batches):
            image_batches.append(list_images[batch_id * batch_size: batch_id * batch_size + batch_size])
        return image_batches


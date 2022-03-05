import torch
import  torch.nn.functional as F
import numpy as np 
class Topk:
    def __init__(self, topk=1, class_id_mapping=None):
        self.topk = topk 
        self.class_id_mapping = class_id_mapping 
        if self.class_id_mapping is not None:
            self.id_class_mapping = self._parse_id_class_mapping()

    def __call__(self, x, filenames=None, multiple_label=False):
        """

        Args:
            x (tensor): shape BxC
            multiple_label (bool, optional): multi-label classification ~ more than 
            one right answer -> sigmoid
                multi-class ~ only one right answer -> softmax

            
            . Defaults to False.
        """
        assert isinstance(x, torch.Tensor)

        x = F.sigmoid(x) if multiple_label else F.softmax(x, dim=-1)
        x = x.detach().cpu().numpy()
        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype('int32') \
                        if not multiple_label else np.where(probs >= 0.5)[0].astype('int32')
            class_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                class_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_mapping is not None:
                    label_name_list.append(self.id_class_mapping[i.item()])

            result = {
                'class_ids': class_id_list,
                'scores': np.round(score_list, decimals=4).tolist()

            }
            if filenames is not None:
                result['file_name'] = filenames[idx]
            if label_name_list is not None:
                result['label_name_list'] = label_name_list
            y.append(result)
        return y
    
    def _parse_id_class_mapping(self):
        id_class_mapping = {}
        for class_name, id in self.class_id_mapping.items():
            id_class_mapping[id] = class_name

        return id_class_mapping

import cv2
import numpy as np

class DecodeImage:
    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb 

    def __call__(self, img):
        assert isinstance(img, np.ndarray) and img is not None, "invalid img in DecodeImage"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img 

class Resize:
    def __init__(self, size, keep_ratio) -> None:
        """

        Args:
            size (tuple): desired size HxW
            keep_ratio (bool): whether keep ratio size image
        """
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, img):
        """ 

        Args:
            img (np.ndarray): input image
        """
        # from IPython import embed; embed()
        assert img is not None, 'image must be not None'
        ori_h, ori_w = img.shape[:2]
        target_h, target_w = self.size 

        if not self.keep_ratio:
            img = cv2.resize(img, (target_w, target_h), cv2.INTER_AREA)
        else:
            w_keep_ratio = int(target_h * float(ori_w) / float(ori_h))
            if w_keep_ratio >= target_w:
                img = cv2.resize(img, (target_w, target_h), cv2.INTER_AREA)
            else:
                img = cv2.resize(img, (w_keep_ratio, target_h), cv2.INTER_AREA)
                padding = target_w - w_keep_ratio
                img = cv2.copyMakeBorder(img, 0, 0, 0, padding, borderType=cv2.BORDER_CONSTANT, value=255)
        return img 



class Normalize:
    def __init__(self, scale, std, mean):
        if isinstance(scale, str):
            self.scale = eval(scale)
        else: 
            self.scale = scale
        self.std = np.array(std).reshape((1, 1, 3)).astype('float32')
        self.mean = np.array(mean).reshape((1, 1, 3)).astype('float32')

    def __call__(self, img):
        assert img is not None, 'image must be not None'
        assert isinstance(img, np.ndarray), "invalid input in Normalize"
        img = (img.astype('float32') * self.scale - self.mean) / self.std

        return img 
        
class DeNormalize:
    def __init__(self, scale, std, mean):
        if isinstance(scale, str):
            self.scale = eval(scale)  # 1.0/255.0
        else: 
            self.scale = scale
        self.std = np.array(std).reshape((1, 1, 3)).astype('float32')
        self.mean = np.array(mean).reshape((1, 1, 3)).astype('float32')

    def __call__(self, img):
        assert isinstance(img, np.ndarray), "invalid input in Normalize"
        img = (img.astype('float32') * self.std + self.mean) / self.scase
        return img 
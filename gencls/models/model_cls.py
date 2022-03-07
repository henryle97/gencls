from turtle import forward
import torch.nn as nn 
import torch 

from gencls.models.backbones.mobilenet_v3 import MobileNetV3
from gencls.models.heads.cls_head import ClsHead

class ClsModel(nn.Module):
    def __init__(self, num_classes, backbone_config):
        super(ClsModel, self).__init__()
        if backbone_config['type'] == 'mobilenetv3':
            self.feature_extractor = MobileNetV3(**backbone_config['hyperparameter'])
        else:
            raise NotImplementedError(f"{backbone_config['type']} not implemented!!!")

        self.head = ClsHead(in_channels=self.feature_extractor.out_channels,
                            out_channels=num_classes)

        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(
        #     in_features=self.feature_extractor.out_channels,
        #     out_features=num_classes,
        #     bias=True
        # )

    def forward(self, x):
        features = self.feat(x)
        outs = self.head(features)
        return outs

    # def forward(self, x):
    #     x = self.feature_extractor(x)
    #     x = self.pool(x)
    #     x = torch.reshape(x, shape=[x.shape[0], x.shape[1]])
    #     x = self.fc(x)
    #     return x



        
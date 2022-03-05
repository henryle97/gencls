from turtle import forward
import torch.nn as nn 



class ClsHead(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                ):
        """ClsHead

        Args:
            in_channels (int): _description_
            out_channels (_type_): _description_
        """
        super(ClsHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            bias=True
        )

    def forward(self, inputs): 
        x = self.gap(inputs)
        x = x.view(inputs.size(0), -1)
        x = self.fc(x)
        return x
from typing import List
import torch.nn as nn


class BaseBackbone(nn.Module):
    """
    Make feature map from input image
    """
    def __init__(self,
                 dep_mul: float = 1.0,
                 wid_mul: float = 1.0,
                 out_features: List[str] = None):
        super().__init__()
        self.dep_mul = dep_mul
        self.wid_mul = wid_mul
        self.out_features = out_features

    """
    def forward(self, x):
        input: 
            images(bs, c, h, w)
        
        output :
            list of feature map / last feature map
            
        outputs = {}
        x = self.conv1(x)
        outputs["f1"] = x
        x = self.conv2(x)
        outputs["f2"] = x
        ...
        if self.out_features is not None:
            return {k: v for k, v in outputs.items() if k in self.out_features}
        else:
            return x
    """
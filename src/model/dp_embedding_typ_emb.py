from numpy.core.fromnumeric import std
import torch
import torch.nn as nn
from torch.nn.init import normal_ as normal
import numpy as np
from typing import List
'''
# logging and o ur extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.DeepMD')

def dump(msg, *args, **kwargs):
    logger.log(logging_level_DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(logging_level_SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)
'''

class LayerModule(nn.Module):
    def __init__(self, 
                 weight: torch.Tensor, 
                 bias: torch.Tensor = None, 
                 resnet_dt: torch.Tensor = None):
        super(LayerModule, self).__init__()
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.bias = nn.Parameter(bias, requires_grad=True) if bias is not None else None
        self.resnet_dt = nn.Parameter(resnet_dt, requires_grad=True) if resnet_dt is not None else None
'''
description: 
The reason for merging the code of type_embedding_net and embedding_net here is that 
only the first layer has a different input size. 
In the embedding_net, the input is S_ij and is set to 1.
In the type_embedding_net, the input scale is depend on type_feature_num
return {*}
author: wuxingxing
'''
class EmbeddingNet(nn.Module):
    '''
    description: 
        when it's type_embedding_net: the type_feat_num is None, the is_type_emb is True
        when it's embedding_net: the type_feat_num is out layer of type_embedding_net and the is_type_emb is False
    param {*} self
    param {*} cfg
    param {*} type_feat_num
    param {*} is_type_emb
    return {*}
    author: wuxingxing
    '''    
    def __init__(self, 
                 network_size: List[int],
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str,
                 type_feat_num: int = None,
                 is_type_emb: bool = False):
        super(EmbeddingNet, self).__init__()
        if is_type_emb == True:
            self.network_size = network_size
        else:
            self.network_size = [1 + type_feat_num] + network_size
        self.bias_flag = bias
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None

        self.layers = nn.ModuleList()
        
        # initial weight normalization
        for i in range(1, len(self.network_size)):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))

            bias = None
            if self.bias_flag:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)

            resnet_dt = None
            if self.resnet_dt_flag:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=1, std=0.001)

            self.layers.append(LayerModule(wij, bias, resnet_dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if self.bias_flag and layer.bias is not None:
                hiden = torch.matmul(x, layer.weight) + layer.bias
            else:
                hiden = torch.matmul(x, layer.weight)

            hiden = self.activation(hiden) 

            if self.network_size[i+1] == self.network_size[i]:
                if self.resnet_dt_flag and layer.resnet_dt is not None:
                    x = hiden * layer.resnet_dt + x
                else:
                    x = hiden + x
            elif self.network_size[i+1] == 2 * self.network_size[i]:
                if self.resnet_dt_flag and layer.resnet_dt is not None:
                    x = torch.cat((x, x), dim=-1) + hiden * layer.resnet_dt
                else:
                    x = torch.cat((x, x), dim=-1) + hiden
            else:
                x = hiden
        return x

class FittingNet(nn.Module):

    def __init__(self, 
                 network_size: List[int],
                 bias: bool,
                 resnet_dt: bool,
                 activation: str,
                 input_dim: int,
                 ener_shift: float, 
                 magic: bool = False):
        super(FittingNet, self).__init__()
        self.network_size = [input_dim] + network_size
        self.bias_flag = bias
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None

        self.layers = nn.ModuleList()

        for i in range(1, len(self.network_size)-1):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))

            bias = None
            if self.bias_flag:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)

            resnet_dt = None
            if i > 1 and self.resnet_dt_flag:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=0.1, std=0.001)

            self.layers.append(LayerModule(wij, bias, resnet_dt))

        i = len(self.network_size) - 1
        wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
        normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))

        if self.bias_flag:
            bias_init = torch.Tensor(1, self.network_size[i])
            normal(bias_init, mean=ener_shift, std=1.0)

        self.layers.append(LayerModule(wij, bias_init, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:    # 对于非最后一层
                if self.bias_flag and layer.bias is not None:
                    hiden = torch.matmul(x, layer.weight) + layer.bias
                else:
                    hiden = torch.matmul(x, layer.weight)
                hiden = self.activation(hiden)
                if i > 0:
                    if self.network_size[i+1] == self.network_size[i] and layer.resnet_dt is not None:
                        x = hiden * layer.resnet_dt + x
                    else:
                        x = hiden + x
                else:
                    x = hiden

        for i, layer in enumerate(self.layers):
            if i == len(self.network_size) - 2:
                if self.bias_flag:
                    x = torch.matmul(x, layer.weight) + layer.bias
                else:
                    x = torch.matmul(x, layer.weight)
        return x                                     
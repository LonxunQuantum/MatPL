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

class EmbeddingNet(nn.Module):
    def __init__(self, 
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str, 
                 magic = False):
        super(EmbeddingNet, self).__init__()
        self.network_size = [1] + network_size
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
        # print()

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
                    x = torch.cat((x, x), dim=-1)  + hiden * layer.resnet_dt
                else:
                    x = torch.cat((x, x), dim=-1)  + hiden
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
                 magic = False):
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
        wij = torch.randn(self.network_size[i-1], self.network_size[i])
        normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))

        if self.bias_flag:
            bias_init = torch.randn(1, self.network_size[i])
            normal(bias_init, mean=ener_shift, std=1.0)
        
        self.layers.append(LayerModule(wij, bias_init, None))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:        # 对于非最后一层
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
'''    
class EmbeddingNet0(nn.Module):
    def __init__(self, 
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str,
                 magic = False):
        super(EmbeddingNet, self).__init__()
        self.network_size = [1] + network_size
        self.bias = bias
        self.resnet_dt = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None
        self.linears = nn.ModuleList()
        self.resnet = nn.ParameterList()

        # initial weight normalization
        for i in range(len(self.network_size) - 1):
            self.linears.append(nn.Linear(self.network_size[i], self.network_size[i+1], bias=self.bias))
            if self.bias:
                nn.init.normal_(self.linears[i].bias, mean=0.0, std=1.0)
            if self.network_size[i] == self.network_size[i+1] or self.network_size[i] * 2 == self.network_size[i+1]:
                resnet_tensor = torch.Tensor(1, self.network_size[i+1])
                nn.init.normal_(resnet_tensor, mean=1, std=0.001)
                self.resnet.append(nn.Parameter(resnet_tensor, requires_grad=True))
            nn.init.normal_(self.linears[i].weight, mean=0, std=(1.0 / np.sqrt(self.network_size[i] + self.network_size[i+1])))
        # print()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 0
        for i, linear in enumerate(self.linears):
            hiden = linear(x)
            hiden = self.activation(hiden)
            if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                for ii, resnet in enumerate(self.resnet):
                    if ii == m:
                        x = hiden * resnet + x
                m += 1
            elif self.network_size[i] == self.network_size[i+1] and (not self.resnet_dt):
                x = hiden + x
            elif self.network_size[i] * 2 == self.network_size[i+1] and self.resnet_dt:
                for ii, resnet in enumerate(self.resnet):
                    if ii == m:
                        x = hiden * resnet + torch.cat((x, x), dim=-1)
                m += 1
            elif self.network_size[i] * 2 == self.network_size[i+1] and (not self.resnet_dt):
                x = hiden + torch.cat((x, x), dim=-1)
            else:
                x = hiden
        return x

class FittingNet0(nn.Module):

    def __init__(self,  
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str,
                 input_dim: int, 
                 ener_shift: float, 
                 magic = False):
        super(FittingNet, self).__init__()
        self.network_size = [input_dim] + network_size
        self.bias = bias
        self.resnet_dt = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None
        self.linears = nn.ModuleList()
        self.resnet = nn.ParameterList()

        for i in range(len(self.network_size)-1):
            if i == len(self.network_size) - 2:
                self.linears.append(nn.Linear(self.network_size[i], self.network_size[i+1], bias=True))
                nn.init.normal_(self.linears[i].bias, mean=ener_shift, std=1.0)
            else:
                self.linears.append(nn.Linear(self.network_size[i], self.network_size[i+1], bias=self.bias))
                if self.bias:
                    nn.init.normal_(self.linears[i].bias, mean=0.0, std=1.0)
            if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                resnet_tensor = torch.Tensor(1, self.network_size[i+1])
                nn.init.normal_(resnet_tensor, mean=0.1, std=0.001)
                self.resnet.append(nn.Parameter(resnet_tensor, requires_grad=True))
            nn.init.normal_(self.linears[i].weight, mean=0, std=(1.0 / np.sqrt(self.network_size[i] + self.network_size[i+1])))
        # print()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 0
        for i, linear in enumerate(self.linears):
            if i == len(self.network_size) - 2:
                hiden = linear(x)
                x = hiden
            else:
                hiden = linear(x)
                hiden = self.activation(hiden)
                if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hiden * resnet + x
                    m += 1
                elif self.network_size[i] == self.network_size[i+1] and (not self.resnet_dt):
                    x = hiden + x
                elif self.network_size[i] * 2 == self.network_size[i+1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hiden * resnet + torch.cat((x, x), dim=-1)
                    m += 1
                elif self.network_size[i] * 2 == self.network_size[i+1] and (not self.resnet_dt):
                    x = hiden + torch.cat((x, x), dim=-1)
                else:
                    x = hiden
        return x
'''
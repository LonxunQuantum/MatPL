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

class FittingNet(nn.Module):

    def __init__(self,  
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str, 
                 input_dim: int, 
                 ener_shift: float, 
                 magic = False,
                 nep_txt_param: List[np.array]=None,
                 last_bias:bool=True):
        super(FittingNet, self).__init__()
        self.network_size = [input_dim] + network_size
        self.bias_flag = bias
        self.last_bias = last_bias #if last_bias is false, the common_bias will used, just like gpumd, the test show that common_bias have lower accuracy
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None

        self.layers = nn.ModuleList()
        from_nep_txt = False
        if nep_txt_param is not None:
            from_nep_txt = True
        for i in range(1, len(self.network_size)-1):
            if from_nep_txt:
                wij = torch.from_numpy(nep_txt_param[0])
                bias= torch.from_numpy(nep_txt_param[1])
            else:
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
        if from_nep_txt:
            wij = torch.from_numpy(nep_txt_param[2])
            bias_value = torch.as_tensor(nep_txt_param[3], dtype=wij.dtype)
            if bias_value.ndim == 0:
                bias_init = bias_value * torch.ones(1, self.network_size[i], dtype=wij.dtype)
            else:
                bias_init = bias_value.reshape(1, self.network_size[i]).to(dtype=wij.dtype)
        else:
            wij = torch.randn(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))

            if self.last_bias:
                bias_init = torch.randn(1, self.network_size[i])
                normal(bias_init, mean=0.0, std=1.0)
                ener_shift_tensor = torch.as_tensor(ener_shift, dtype=bias_init.dtype)
                if ener_shift_tensor.ndim == 0:
                    bias_init += ener_shift_tensor
                else:
                    bias_init += ener_shift_tensor.reshape(1, self.network_size[i])
        
        if self.last_bias:
            self.layers.append(LayerModule(wij, bias_init, None))
        else:
            self.layers.append(LayerModule(wij, None, None))
    
    '''
    description: 
        get all nn params to nep.txt
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_param_list(self):
        param_list = []
        last_bias_list = []
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:        # 对于非最后一层
                if self.bias_flag and layer.bias is not None:
                    # hiden = torch.matmul(x, layer.weight) + layer.bias
                    param_list.extend(list(layer.weight.transpose(1, 0).flatten().cpu().detach().numpy())) # the nep only has one hidden layer, shape is [35,100], convert to[100, 35]
                    param_list.extend((-layer.bias).flatten().cpu().detach().numpy()) # the nep use tanh(x*w - b)
                else:
                    param_list.extend(list(layer.weight.flatten().cpu().detach().numpy()))
                if i > 0:
                    if self.network_size[i+1] == self.network_size[i] and layer.resnet_dt is not None:
                        param_list.extend(list(layer.resnet_dt.flatten().cpu().detach().numpy()))
                    else:
                        pass
                else:
                    pass

        for i, layer in enumerate(self.layers):
            if i == len(self.network_size) - 2:
                if self.last_bias:
                    param_list.extend(list(layer.weight.flatten().cpu().detach().numpy())) # the nep use tanh(x*w - b)
                    last_bias_list.extend((-layer.bias).flatten().cpu().detach().numpy())
                else:
                    param_list.extend(list(layer.weight.flatten().cpu().detach().numpy()))
        return param_list, last_bias_list



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
                if self.last_bias:
                    x = torch.matmul(x, layer.weight) + layer.bias #
                else:
                    x = torch.matmul(x, layer.weight)
        return x


class QNEPFittingNet(nn.Module):

    def __init__(self,
                 network_size: List[int],
                 bias: bool,
                 resnet_dt: bool,
                 activation: str,
                 input_dim: int,
                 ener_shift: float,
                 charge_mode: int,
                 magic = False,
                 nep_txt_param: List[np.array]=None,
                 last_bias:bool=True):
        super(QNEPFittingNet, self).__init__()
        self.network_size = [input_dim] + network_size
        self.bias_flag = bias
        self.last_bias = last_bias
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None
        self.charge_mode = charge_mode
        self.output_num = 2

        self.layers = nn.ModuleList()
        from_nep_txt = nep_txt_param is not None
        for i in range(1, len(self.network_size)):
            if from_nep_txt:
                wij = torch.from_numpy(nep_txt_param[0])
                layer_bias = torch.from_numpy(nep_txt_param[1])
            else:
                wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
                normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))

                layer_bias = None
                if self.bias_flag:
                    layer_bias = torch.Tensor(1, self.network_size[i])
                    normal(layer_bias, mean=0, std=1)

            resnet_dt = None
            if i > 1 and self.resnet_dt_flag:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=0.1, std=0.001)

            self.layers.append(LayerModule(wij, layer_bias, resnet_dt))

        hidden_dim = self.network_size[-1]
        if from_nep_txt:
            output_weight = torch.from_numpy(nep_txt_param[2])
            if output_weight.ndim == 1:
                output_weight = output_weight.reshape(hidden_dim, self.output_num)
            energy_weight = output_weight[:, 0:1]
            charge_weight = output_weight[:, 1:2]
            energy_bias = None
            if self.last_bias:
                energy_bias = torch.as_tensor(nep_txt_param[3], dtype=energy_weight.dtype)
                if energy_bias.ndim > 0:
                    energy_bias = energy_bias.reshape(-1)[0]
                energy_bias = energy_bias.reshape(1, 1)
        else:
            energy_weight = torch.Tensor(hidden_dim, 1)
            charge_weight = torch.Tensor(hidden_dim, 1)
            normal(energy_weight, mean=0, std=(1.0 / np.sqrt(hidden_dim + 1)))
            normal(charge_weight, mean=0, std=(1.0 / np.sqrt(hidden_dim + 1)))
            energy_bias = None
            if self.last_bias:
                energy_bias = torch.as_tensor(ener_shift, dtype=energy_weight.dtype).reshape(1, 1)

        self.energy_head = LayerModule(energy_weight, energy_bias, None)
        self.charge_head = LayerModule(charge_weight, None, None)

    def get_param_list(self):
        param_list = []
        last_bias_list = []
        for i, layer in enumerate(self.layers):
            if self.bias_flag and layer.bias is not None:
                param_list.extend(list(layer.weight.transpose(1, 0).flatten().cpu().detach().numpy()))
                param_list.extend((-layer.bias).flatten().cpu().detach().numpy())
            else:
                param_list.extend(list(layer.weight.flatten().cpu().detach().numpy()))
            if i > 0:
                if self.network_size[i+1] == self.network_size[i] and layer.resnet_dt is not None:
                    param_list.extend(list(layer.resnet_dt.flatten().cpu().detach().numpy()))

        param_list.extend(list(self.energy_head.weight.flatten().cpu().detach().numpy()))
        if self.energy_head.bias is not None:
            last_bias_list.extend((-self.energy_head.bias).flatten().cpu().detach().numpy())
        param_list.extend(list(self.charge_head.weight.flatten().cpu().detach().numpy()))
        return param_list, last_bias_list

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
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

        energy = torch.matmul(x, self.energy_head.weight)
        if self.energy_head.bias is not None:
            energy = energy + self.energy_head.bias
        charge = torch.matmul(x, self.charge_head.weight)
        return energy, charge
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

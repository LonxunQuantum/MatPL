#
# Here we implement some bi-direction dmirror layers as the base
# components to build dmirror style networks. The main idea is,
# each layer instance can handle both the "forward" and "d_forward"
# operation, and it collects all the required data for "d_forward"
# phase during the "forward" phase. By doing so, we can implement
# a hidden style weight & variable sharing, we just need to describe
# the network and all the sharing issues will be done automaticlly
#
# Basic dmirror layers:
#
#   dmirror_linear(self, in_dim, out_dim, bias=True)
#       * similar to nn.linear
#
#   dmirror_activation(self, func, d_func)
#       * a general activation layer
#       * just specify your activation/d_activation function 
#
#   dmirror_scale()
#       * automaticlly scale/normalize input features
#       * not implemented yet
#
#   dmirror_layer_norm()
#   dmirror_batch_norm()
#       * not implemented yet
#
# Basic dmirror network modules:
#
#   dmirror_FC(self, cfg, act_func, d_act_func)
#       * cfg describes the base part of a dmirror style network,
#         the whole network (including the mirrored part) will be
#         automaticlly generated. An example:
#
#           cfg = [
#                   (scale,),                   # layer: scale_1
#                   (linear, 16, 32, True),     # layer: linear_1, bias=True
#                   (activation,),              # layer: activation_1
#                   (linear, 32, 64, True),     # layer: linear_2, bias=True
#                   (activation,),              # layer: activation_2
#                   (linear, 64, 8, False),     # layer: linear3, bias=False
#                   (activation,),              # layer: activation_3
#                   (linear, 8, 1, False),      # layer: linear4, bias=False
#           ]
#
#           the auto generated (virtual) mirrored part should be:
#           [
#                   (d_linear, 1, 8, False),    # layer: linear4, bias=False
#                   (d_activation,),            # layer: activation_3
#                   (d_linear, 8, 64, False),   # layer: linear3, bias=False
#                   (d_activation,),            # layer: activation_2
#                   (d_linear, 64, 32, True),   # layer: linear_2, bias=False
#                   (d_activation,),            # layer: activation_1
#                   (d_linear, 32, 16, True),   # layer: linear_1, bias=False
#                   (d_scale,),                 # layer: scale_1
#           ]
#
#       * act_func/d_act_func is the actual activation/d_activation
#         function which is called by the dmirror_activation layer
#       * forward() returns the last layer's output of both the base
#         part and the mirrored part, the method call should like:
#           y1, y2 = instance_dimirror_FC.forward(x)
#         or
#           Ei, dEi_dFeat = MLFF_dmirror_FC.forward(Feat_atom_i)
#

import torch
import torch.nn as nn
import collections

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.dmirror')

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

# dmirror implementation
#
class dmirror_linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, magic=False):
        super(dmirror_linear, self).__init__()
        self.bias = None

        # for pesudo random number generator for float64/float32 precision test
        self.rand_a = 25214903917
        self.rand_c = 11
        self.rand_p = 2021

        if (magic == False):
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=True)
            if (bias == True):
                self.bias = nn.Parameter(torch.randn(out_dim), requires_grad=True)
        else:
            warmup_my_rand = self.my_rand_2d(out_dim, in_dim)
            self.weight = nn.Parameter(self.my_rand_2d(out_dim, in_dim), requires_grad=True)
            if (bias == True):
                self.bias = nn.Parameter(self.my_rand_1d(out_dim), requires_grad=True)

    # random number generator, maybe their better place is train.py
    def my_rand_core(self):
        r = (self.rand_a * self.rand_p + self.rand_c) % 10000
        self.rand_p = r
        return r

    def my_rand_2d(self, m, n):
        res = torch.randn(m, n)
        for i in range(m):
            for j in range(n):
                res[i, j] = float(self.my_rand_core() / 10000.0)
        return res

    def my_rand_1d(self, m):
        res = torch.randn(m)
        for i in range(m):
            res[i] = float(self.my_rand_core() / 10000.0)
        return res

    def forward(self, x):
        if (self.bias is not None):
            return torch.matmul(x, self.weight.t()) + self.bias
        else:
            return torch.matmul(x, self.weight.t())

    def forward_r(self, x):
        return torch.matmul(x, self.weight)


class dmirror_activation(nn.Module):
    def __init__(self, func, d_func):
        super(dmirror_activation, self).__init__()
        self.func = func
        self.d_func = d_func
        self.k = torch.tensor([])

    def forward(self, x):
        self.k = x
        return self.func(x)

    def forward_r(self, x):
        return x * self.d_func(self.k)


class dmirror_FC(nn.Module):
    def __init__(self, cfg, act_func, d_act_func, magic=False):
        super(dmirror_FC, self).__init__()
        self.cfg = cfg
        self.act_func = act_func
        self.d_act_func = d_act_func
        self.layers = []
        self.layers_r = []

        # parse cfg & generating layers
        #
        idx_linear = 1
        idx_activation = 1
        for idx, item in enumerate(self.cfg):
            layer_type = item[0]
            if (layer_type == 'linear'):
                in_dim = item[1]
                out_dim = item[2]
                bias = item[3]
                self.layers.append((
                    'dmirror_linear_'+str(idx_linear),
                    dmirror_linear(in_dim, out_dim, bias, magic)
                ))
                idx_linear += 1
            elif (layer_type == 'activation'):
                self.layers.append((
                    'dmirror_activation_'+str(idx_activation),
                    dmirror_activation(act_func, d_act_func)
                ))
                idx_activation += 1
            elif (layer_type == 'scale'):
                raise RuntimeError(
                    "Notimplemented for layer_type = %s" %(layer_type)
                )
            else:
                raise ValueError(
                    "Invalid for layer_type = %s" %(layer_type)
                )
        self.layers_r = list(reversed(self.layers))

        # the layer parameters will be registered to nn Module,
        # so optimizer can update the layer parameters.
        #
        self.base_net = nn.Sequential(
            collections.OrderedDict(self.layers)
        )
        self.mirror_net = nn.Sequential(
            collections.OrderedDict(self.layers_r)
        )
        info("dmirror_FC: start of network instance dump ==============>")
        info("<----------------------- base_net ----------------------->")
        info(self.base_net)
        info("<---------------------- mirror_net ---------------------->")
        info(self.mirror_net)
        info("dmirror_FC: end of network instance dump ================>")

    # we can't call forward() of sequentialized module, since
    # we extened the param list of the layers' forward()
    #
    def forward(self, x):
        for name, obj in (self.layers):
            x = obj.forward(x)
        res0 = x

        x = torch.ones_like(res0)
        for name, obj in (self.layers_r):
            x = obj.forward_r(x)
        res1 = x

        return res0, res1


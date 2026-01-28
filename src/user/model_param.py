from utils.json_operation import get_parameter, get_required_parameter
from src.user.nep_param import NepParam
class NetParam(object):
    def __init__(self, net_type:str) -> None:
        self.net_type = net_type

    def set_params(self, network_size: list, bias:bool, resnet_dt:bool, activation:str, physical_property:list=None):
        self.physical_property = physical_property
        self.network_size = network_size
        self.bias = bias
        self.resnet_dt = resnet_dt
        self.activation=activation      
        
    def to_dict(self):
        dicts = {}
        if self.network_size is not None:
            dicts["network_size"] = self.network_size
        else:
            dicts["network_size"] = []
        if "type_" in self.net_type:
            dicts["physical_property"] = self.physical_property
        dicts["bias"] = self.bias
        dicts["resnet_dt"] = self. resnet_dt 
        dicts["activation"] = self.activation
        return dicts
    
    def to_dict_std(self):
        dicts = {}
        if self.network_size is not None:
            dicts["network_size"] = self.network_size
        if "type_" in self.net_type:
            dicts["physical_property"] = self.physical_property
        #dicts["bias"] = self.bias, 
        # if self.resnet_dt is False:
        #     dicts["resnet_dt"] = self.resnet_dt
        #dicts["activation"] = self.activation
        return dicts

class ModelParam(object):
    def __init__(self) -> None:
        self.type_embedding_net = None
        self.embedding_net = None
        self.fitting_net = None
        self.nep_param:NepParam = None

    '''
    description: 
        if type_embedding in first layer of json file is True:
           if type_embedding is not set under layer of model/descriptor, then, the default type embedding set will be used
        else:
           if type_embedding is set under layer of model/descriptor, then the type embedding set will be used
    param {*} self
    param {bool} type_embedding
    param {dict} descriptor_dict
    return {*}
    author: wuxingxing
    '''
    def set_type_embedding_net(self, network_size:list, bias:bool, resnet_dt:bool, activation:str, physical_property:list):
        self.type_embedding_net = NetParam("type_embedding_net")
        self.type_embedding_net.set_params(network_size, bias, resnet_dt, activation, physical_property)
    
    def set_embedding_net(self, network_size:list, bias:bool, resnet_dt:bool, activation:str):
        self.embedding_net = NetParam("embedding_net")
        self.embedding_net.set_params(network_size, bias, resnet_dt, activation)

    def set_dp_fitting_net(self, fitting_net_dict:dict):
        # fitting_net_dict = get_parameter("fitting_net",json_input, {})
        network_size = get_parameter("network_size", fitting_net_dict, [50, 50, 50, 1])
        if network_size[-1] != 1:
            raise Exception("Error: The last layer of the fitting network should have a size of 1 for etot energy, but the input size is {}!".format(network_size[-1]))
        bias = get_parameter("bias", fitting_net_dict, True)
        resnet_dt = get_parameter("resnet_dt", fitting_net_dict, True)
        activation = get_parameter("activation", fitting_net_dict, "tanh")
        self.fitting_net = NetParam("fitting_net")
        self.fitting_net.set_params(network_size, bias, resnet_dt, activation)

    def set_nn_fitting_net(self, fitting_net_dict:dict):
        # fitting_net_dict = get_parameter("fitting_net",json_input, {})
        network_size = get_parameter("network_size", fitting_net_dict,[15,15,1])
        if not isinstance(network_size, list):
            network_size = [network_size]
        if len(network_size) > 1 and network_size[-1] != 1:
            raise Exception("Error: The last layer of the fitting network should have a size of 1 for etot energy, but the input size is {}!".format(network_size[-1]))
        bias = True # get_parameter("bias", fitting_net_dict, True)
        resnet_dt = False # get_parameter("resnet_dt", fitting_net_dict, False)
        activation = "tanh" #get_parameter("activation", fitting_net_dict, )
        self.fitting_net = NetParam("fitting_net")
        self.fitting_net.set_params(network_size, bias, resnet_dt, activation)

    # def to_dict(self):
    #     # dicts = {}
    #     # if self.embedding_net is not None:
    #     #     dicts[self.embedding_net.net_type] = self.embedding_net.to_dict()
    #     # if self.fitting_net is not None:
    #     #     dicts[self.fitting_net.net_type] = self.fitting_net.to_dict()
    #     return self.fitting_net.to_dict_std()



from utils.json_operation import get_parameter, get_required_parameter
from src.user.nep_param import NepParam

class OptimizerParam(object):
    def __init__(self) -> None:
        pass

    def set_optimizer(self, json_source:dict, nep_param:NepParam=None):
        optimizer_dict = get_parameter("optimizer", json_source, {})
        self.opt_name = get_parameter("optimizer", optimizer_dict, "ADAM")
        self.batch_size = get_parameter("batch_size", optimizer_dict, 1)
        # self.batch_size = 1
        self.epochs = get_parameter("epochs", optimizer_dict, 30)
        self.print_freq = get_parameter("print_freq", optimizer_dict, 10)
        # the start epoch could be reset at the resume model code block
        self.reset_epoch = get_parameter("reset_epoch", optimizer_dict, True)
        self.start_epoch = get_parameter("start_epoch", optimizer_dict, 1)

        self.max_norm    = get_parameter("max_norm", optimizer_dict, None)
        self.norm_type   = get_parameter("norm_type", optimizer_dict, None)
        self.clip_value  = get_parameter("clip_value", optimizer_dict, None)
        if self.clip_value is not None and self.max_norm is not None:
            raise Exception("ERROR! 'max_norm' and 'norm_type' for Norm clipping and 'clip_value' for value clipping cannot be set simultaneously.!")
        if self.max_norm is not None:
            if self.norm_type is None:
                self.norm_type = 2
            else:
                if self.norm_type not in [1, 2]:
                    raise Exception("ERROR! the input norm_type only could be set as '1' for L1 or '2' for L2!")
        
        # learining rate,  https://zhuanlan.zhihu.com/p/261134624
        self.t_0    = get_parameter("t_0", optimizer_dict, None)
        self.t_mult   = get_parameter("t_mult", optimizer_dict, None)
        if (self.t_0 is None and self.t_mult is not None) or (self.t_0 is not None and self.t_mult is None):
            raise Exception("ERROR! the input t_0 and t_mult need to be set simultaneously!")
        self.verbose  = get_parameter("verbose", optimizer_dict, False) # nouse

        self.lambda_1 = None
        if "lambda_1" in optimizer_dict:
            raise Warning("The lambda_1 is not realized now!")
            # self.lambda_1 = get_parameter("lambda_1", optimizer_dict, None) # weight of regularization term
            # if self.lambda_1 is not None and self.lambda_1 < 0:
            #     raise Exception("ERROR! the lambda_1 should >= 0 !")

        self.lambda_2 = get_parameter("lambda_2", optimizer_dict, None) # weight of norm regularization term
        if self.lambda_2 is not None and self.lambda_2 < 0:
            raise Exception("ERROR! the lambda_2 should >= 0 !")

        if "KF" in self.opt_name.upper():  #set Kalman Filter Optimizer params
            self.kalman_lambda = get_parameter("kalman_lambda", optimizer_dict, 0.98)
            self.kalman_nue = get_parameter("kalman_nue", optimizer_dict, 0.9987)
            self.block_size = get_parameter("block_size", optimizer_dict, 5120)
            self.nselect = get_parameter("nselect", optimizer_dict, 24)
            self.groupsize = get_parameter("groupsize", optimizer_dict, 6)
            self.p0_weight = get_parameter("p0_weight", optimizer_dict, 0.01)
            if self.p0_weight > 1.0:
                raise Exception("ERROR! the p0_weight must be less than 1.0, with 0.01 as default value. If set to 1, it means regularization is not applicable.")

        elif "ADAM" in self.opt_name.upper():   # set ADAM Optimizer params
            self.learning_rate = get_parameter("learning_rate", optimizer_dict, 0.001)
            self.weight_decay = get_parameter("weight_decay", optimizer_dict, 1e-4)
            self.momentum = get_parameter("momentum", optimizer_dict, 0.9)
            self.gamma = get_parameter("gamma", optimizer_dict, 0.99) # used in nn optimizer
            self.step = get_parameter("step", optimizer_dict, 100) # used in nn optimizer
            self.scheduler = get_parameter("scheduler", optimizer_dict, None) # used in nn optimizer

            self.stop_step = get_parameter("stop_step", optimizer_dict, 1000000)
            self.decay_step = get_parameter("decay_step", optimizer_dict, 5000)
            self.stop_lr = get_parameter("stop_lr",optimizer_dict, 3.51e-8)
            # self.set_adam_sgd_params(learning_rate, weight_decay, momentum,\
            #                                 gamma, step, scheduler, stop_step, decay_step, stop_lr)
        else:
            pass

        self.train_energy = get_parameter("train_energy", optimizer_dict, True) 
        self.train_force = get_parameter("train_force", optimizer_dict, True) 
        self.train_ei = get_parameter("train_ei", optimizer_dict, False) 
        self.train_virial = get_parameter("train_virial", optimizer_dict, False) 
        self.train_egroup = get_parameter("train_egroup", optimizer_dict, False) 

        self.force_delta = None
        self.population = None
        self.generation = None
        self.pre_fac_force = 2.0
        self.pre_fac_etot = 1.0
        self.pre_fac_ei = 1.0
        self.pre_fac_virial = 1.0
        self.pre_fac_egroup = 0.1

        if "KF" in self.opt_name.upper():
            self.pre_fac_force = get_parameter("pre_fac_force", optimizer_dict, 2.0) 
            self.pre_fac_etot = get_parameter("pre_fac_etot", optimizer_dict, 1.0) 
            self.pre_fac_ei = get_parameter("pre_fac_ei", optimizer_dict, 1.0) 
            self.pre_fac_virial = get_parameter("pre_fac_virial", optimizer_dict, 1.0) 
            self.pre_fac_egroup = get_parameter("pre_fac_egroup", optimizer_dict, 0.1) 
            
        elif "ADAM" in self.opt_name.upper():
            self.start_pre_fac_force = get_parameter("start_pre_fac_force", optimizer_dict, 1000) 
            self.start_pre_fac_etot = get_parameter("start_pre_fac_etot", optimizer_dict, 0.02) 
            self.start_pre_fac_ei = get_parameter("start_pre_fac_ei", optimizer_dict, 0.1) 
            self.start_pre_fac_virial = get_parameter("start_pre_fac_virial", optimizer_dict, 50.0) 
            self.start_pre_fac_egroup = get_parameter("start_pre_fac_egroup", optimizer_dict, 0.02) 

            self.end_pre_fac_force = get_parameter("end_pre_fac_force", optimizer_dict, 1.0) 
            self.end_pre_fac_etot = get_parameter("end_pre_fac_etot", optimizer_dict, 1.0) 
            self.end_pre_fac_ei = get_parameter("end_pre_fac_ei", optimizer_dict, 2.0) 
            self.end_pre_fac_virial = get_parameter("end_pre_fac_virial", optimizer_dict, 1.0) 
            self.end_pre_fac_egroup = get_parameter("end_pre_fac_egroup", optimizer_dict, 1.0) 

    def to_linear_dict(self):
        opt_dict = {}
        opt_dict["train_energy"] = self.train_energy
        opt_dict["train_force"] = self.train_force
        # opt_dict["train_ei"] = self.train_ei
        opt_dict["pre_fac_force"] = self.pre_fac_force
        opt_dict["pre_fac_etot"] = self.pre_fac_etot
        # opt_dict["pre_fac_ei"] = self.pre_fac_ei
        return opt_dict
    
    def to_dict(self):
        opt_dict = {}
        opt_dict["optimizer"]=self.opt_name
        # opt_dict["start_epoch"] = self.start_epoch
        opt_dict["epochs"] = self.epochs
        opt_dict["batch_size"] = self.batch_size
        opt_dict["print_freq"] = self.print_freq
        if self.lambda_1 is not None:
            opt_dict["lambda_1"] =  self.lambda_1
        if self.lambda_2 is not None:  
            opt_dict["lambda_2"] =  self.lambda_2

        if "KF" in self.opt_name:
            if "LKF" in self.opt_name:
                opt_dict["block_size"] = self.block_size 
            opt_dict["kalman_lambda"] = self.kalman_lambda
            opt_dict["kalman_nue"] = self.kalman_nue

            opt_dict["train_energy"] = self.train_energy
            opt_dict["train_force"] = self.train_force
            # opt_dict["train_ei"] = self.train_ei
            opt_dict["train_virial"] = self.train_virial
            # opt_dict["train_egroup"] = self.train_egroup
    
            opt_dict["pre_fac_force"] = self.pre_fac_force
            opt_dict["pre_fac_etot"] = self.pre_fac_etot
            # opt_dict["pre_fac_ei"] = self.pre_fac_ei
            opt_dict["pre_fac_virial"] = self.pre_fac_virial
            # opt_dict["pre_fac_egroup"] = self.pre_fac_egroup

            opt_dict["p0_weight"] = self.p0_weight

        elif "SGD" in self.opt_name or "ADAM" in self.opt_name:
            if "SGD" in self.opt_name:
                opt_dict["weight_decay"]= self.weight_decay
                opt_dict["momentum"]= self.momentum

            opt_dict["learning_rate"]= self.learning_rate
            opt_dict["stop_lr"] = self.stop_lr
            opt_dict["stop_step"] = self.stop_step
            opt_dict["decay_step"] = self.decay_step

            opt_dict["train_energy"] = self.train_energy
            opt_dict["train_force"] = self.train_force
            # opt_dict["train_ei"] = self.train_ei
            opt_dict["train_virial"] = self.train_virial
            # opt_dict["train_egroup"] = self.train_egroup

            opt_dict["start_pre_fac_force"] = self.start_pre_fac_force
            opt_dict["start_pre_fac_etot"] = self.start_pre_fac_etot
            # opt_dict["start_pre_fac_ei"] = self.start_pre_fac_ei
            opt_dict["start_pre_fac_virial"] = self.start_pre_fac_virial
            # opt_dict["start_pre_fac_egroup"] = self.start_pre_fac_egroup

            opt_dict["end_pre_fac_force"] = self.end_pre_fac_force
            opt_dict["end_pre_fac_etot"] = self.end_pre_fac_etot
            # opt_dict["end_pre_fac_ei"] = self.end_pre_fac_ei
            opt_dict["end_pre_fac_virial"] = self.end_pre_fac_virial
            # opt_dict["end_pre_fac_egroup"] = self.end_pre_fac_egroup

            if self.max_norm is not None:
                opt_dict["max_norm"] = self.max_norm
            if self.norm_type is not None:
                opt_dict["norm_type"] = self.norm_type
            if self.clip_value is not None:
                opt_dict["clip_value"] = self.clip_value
            if self.t_0 is not None:
                opt_dict["t_0"] = self.t_0
            if self.t_mult is not None:
                opt_dict["t_mult"] = self.t_mult

        return opt_dict

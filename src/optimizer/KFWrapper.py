import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import time
import numpy as np
import torch.distributed as dist
import math
import random
from utils.debug_operation import check_cuda_memory
class KFOptimizerWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        atoms_selected: int,
        atoms_per_group: int,
        is_distributed: bool = False,
        distributed_backend: str = "torch",  # torch or horovod
        lambda_l1 = None,
        lambda_l2 = None
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.atoms_selected = atoms_selected  # 24
        self.atoms_per_group = atoms_per_group  # 6
        self.is_distributed = is_distributed
        self.distributed_backend = distributed_backend
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        
    def update_energy(
        self, inputs: list, Etot_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        natoms_sum = None
        if train_type == "DP":
            Etot_predict, _, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                0,
                inputs[4],
                inputs[5],
                is_calc_f=False,
            )
        elif train_type == "NEP":
            Etot_predict, _, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8],
                inputs[9],
                is_calc_f=False,
            )
            natoms_sum = int(inputs[0].shape[0]/inputs[6].shape[0])
        elif train_type == "NN": # nn training
            Etot_predict, _, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                is_calc_f=False,
            )
        elif train_type == "CHEBY":
            Etot_predict, _, _, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                0,
                inputs[5],
                inputs[6]
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))
        # natoms_sum = inputs[2][0, 0]
        if natoms_sum is None:
            natoms_sum = inputs[0].shape[1] # dp or nn model
        self.optimizer.set_grad_prefactor(natoms_sum)

        self.optimizer.zero_grad()
        bs = Etot_label.shape[0]
        error = Etot_label - Etot_predict
        error = error / natoms_sum
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        # if self.is_distributed:
        #     if self.distributed_backend == "horovod":
        #         import horovod as hvd

        #         error = hvd.torch.allreduce(error)
        #     elif self.distributed_backend == "torch":
        #         dist.all_reduce(error)
        #         error /= dist.get_world_size()
        
        _Etot_predict = update_prefactor * Etot_predict
        _Etot_predict[mask] = -1.0 * _Etot_predict[mask]

        _Etot_predict.sum().backward(retain_graph=True)# retain_graph=True is added for nep training
        error = error * math.sqrt(bs)
        #print("Etot steping")
        self.optimizer.step(error)
        return Etot_predict

    def update_egroup(
        self, inputs: list, Egroup_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        natoms_sum = None
        if train_type == "DP":
            
            _, _, _, Egroup_predict, _ = self.model( #dp inputs has 7 para
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                0,
                inputs[4],
                inputs[5],
                is_calc_f=False,
            )
        elif train_type == "NEP":
            _, _, _, Egroup_predict, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8],
                inputs[9],
                is_calc_f=False,
            )
            natoms_sum = int(inputs[0].shape[0]/inputs[6].shape[0])
        elif train_type == "NN": # nn training
            _, _, _, Egroup_predict, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                is_calc_f=False,
            )
        elif train_type == "CHEBY":
            _, _, _, Egroup_predict, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                0,
                inputs[5],
                inputs[6]
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))
        if natoms_sum is None:
            natoms_sum = inputs[0].shape[1]
        self.optimizer.set_grad_prefactor(1.0)

        self.optimizer.zero_grad()
        bs = Egroup_label.shape[0]
        error = Egroup_label - Egroup_predict
        #TODO: as the comment from RuNNer, the error why scaler by atom_num in Etot is because of
        #Etot is the sum of Ei, so maybe we don't need to scaler the egroup error. NEED CHECK!
        error = error 
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        # if self.is_distributed:
        #     if self.distributed_backend == "horovod":
        #         import horovod as hvd

        #         error = hvd.torch.allreduce(error)
        #     elif self.distributed_backend == "torch":
        #         dist.all_reduce(error)
        #         error /= dist.get_world_size()
        
        Egroup_predict = update_prefactor * Egroup_predict
        Egroup_predict[mask] = -1.0 * Egroup_predict[mask]

        Egroup_predict.sum().backward()
        error = error * math.sqrt(bs)
        self.optimizer.step(error)
        return Egroup_predict
    
    def update_virial(
        self, inputs: list, Virial_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        index = [0,1,2,4,5,8]
        if train_type == "NEP":
            data_mask = Virial_label[:, 0] > -1e6
            _Virial_label = Virial_label[:, index][data_mask]
        else:
            data_mask = Virial_label[:, 9] > 0
            _Virial_label = Virial_label[:, index][data_mask]
        natoms_sum = None
        if data_mask.any().item() is False:
            return None
        if train_type == "DP":
            Etot_predict, _, _, _, Virial_predict = self.model(
                inputs[0][data_mask],
                inputs[1],
                inputs[2],
                inputs[3][data_mask],
                0,
                inputs[4],
                inputs[5]
            )
        elif train_type == "NEP":
            Etot_predict, _, _, _, Virial_predict = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8],
                inputs[9]
            )
            natoms_sum = int(inputs[0].shape[0]/inputs[6].shape[0])
        elif train_type == "NN":
            Etot_predict, _, _, Virial_predict = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6]
            )
        elif train_type == "CHEBY":
            Etot_predict, _, _, Virial_predict = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                0,
                inputs[5],
                inputs[6]
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))
        if natoms_sum is None:
            natoms_sum = inputs[0].shape[1]
        self.optimizer.set_grad_prefactor(natoms_sum)
        
        self.optimizer.zero_grad()
        bs = _Virial_label.shape[0]  
        _Virial_predict = Virial_predict[:, index][data_mask]
        error = _Virial_label - _Virial_predict
        error = error / natoms_sum
        mask = error < 0

        # essentially a step length for weight update 
        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        
        error = error.mean()

        _Virial_predict = update_prefactor * _Virial_predict
        _Virial_predict[mask] = -1.0 * _Virial_predict[mask]
        
        _Virial_predict.sum().backward()

        error = error * math.sqrt(bs) 
        
        self.optimizer.step(error)
        return Virial_predict

    def update_egroup_select(
        self, inputs: list, Egroup_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        '''
        A select atoms version for egroup update.
        Base the simply test, it seems like update_egroup is a better choise.
        NEED CHECK!
        '''
        natoms_sum = inputs[3][0, 0]
        #print ("natoms_sum",natoms_sum)
        bs = Egroup_label.shape[0]
        self.optimizer.set_grad_prefactor(self.atoms_per_group)

        index = self.__sample(self.atoms_selected, self.atoms_per_group, natoms_sum)

        for i in range(index.shape[0]):
            self.optimizer.zero_grad()
            _, _, _, Egroup_predict, _ = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
            )
            error_tmp = Egroup_label[:, index[i]] - Egroup_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean()

            # if self.is_distributed:
            #     if self.distributed_backend == "horovod":
            #         import horovod as hvd

            #         error = hvd.torch.allreduce(error)
            #     elif self.distributed_backend == "torch":
            #         dist.all_reduce(error)
            #         error /= dist.get_world_size()

            tmp_egroup_predict = Egroup_predict[:, index[i]] * update_prefactor
            tmp_egroup_predict[mask] = -1.0 * tmp_egroup_predict[mask]

            tmp_egroup_predict.sum().backward()
            error = error * math.sqrt(bs)
            self.optimizer.step(error)
        return Egroup_predict

    def update_force(
        self, inputs: list, Force_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        if train_type == "NEP":
            natoms_sum = int(inputs[0].shape[0]/inputs[6].shape[0])
            bs = inputs[6].shape[0]
            index = self.__sample(self.atoms_selected, self.atoms_per_group, inputs[0].shape[0])
        else:
            natoms_sum = inputs[0].shape[1]
            bs = Force_label.shape[0]
            index = self.__sample(self.atoms_selected, self.atoms_per_group, inputs[0].shape[1])
            
        self.optimizer.set_grad_prefactor(natoms_sum * self.atoms_per_group * 3)

        for i in range(index.shape[0]):
            self.optimizer.zero_grad()
            index_list = index[i]
            if train_type == "DP":
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(
                    inputs[0], inputs[1], inputs[2], inputs[3], 0, inputs[4], inputs[5]
                )
            elif train_type == "NEP":
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], 0, inputs[8], inputs[9]
                )
            elif train_type == "NN":  # nn training
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
                )
            elif train_type == "CHEBY":
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict, dEi_dc = self.model(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 0, inputs[5], inputs[6]
                )
            else:
                raise Exception("Error! the train type {} is not realized!".format(train_type))

            if train_type == "NEP":
                error_tmp = Force_label[index_list] - Force_predict[index_list] # index[i]
                tmp_force_predict = Force_predict[index_list] * update_prefactor

            else:
                error_tmp = Force_label[:, index_list] - Force_predict[:, index_list]
                tmp_force_predict = Force_predict[:, index_list] * update_prefactor
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean() / natoms_sum
            tmp_force_predict[mask] = -1.0 * tmp_force_predict[mask]
            # In order to solve a pytorch bug, reference: https://github.com/pytorch/pytorch/issues/43259
            (tmp_force_predict.sum() + Etot_predict.sum() * 0).backward(retain_graph=True) # retain_graph=True is added for nep training
            error = error * math.sqrt(bs)
            #print("force steping")
            if train_type == "CHEBY":
                self.optimizer.step(error, c_param=self.model.c_param, c_grad=dEi_dc)
            else:
                self.optimizer.step(error)
            # check_cuda_memory(i, i, "update_force index i")
        return Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict

    '''
    description: 
            1. pick out 20% images 
            2. cut it into a several groups 
    return {*}
    author: wuxingxing
    '''
    def update_ei(
        self, inputs: list, Ei_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        if train_type == "DP":
            _, Ei_predict, _, _, _ = self.model( #dp inputs has 7 para
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                0,
                inputs[4],
                inputs[5],
                is_calc_f=False,
            )
        elif train_type == "NEP":
            _, Ei_predict, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                0,
                inputs[8],
                inputs[9],
                is_calc_f=False,
            )
        elif train_type == "NN": # nn training
            _, Ei_predict, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                is_calc_f=False,
            )
        elif train_type == "CHEBY":
            _, Ei_predict, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                0,
                inputs[5],
                inputs[6]
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))

        natoms_sum = inputs[0].shape[1]
        #print ("natoms_sum",natoms_sum)
        bs = Ei_label.shape[0]
        self.optimizer.set_grad_prefactor(1.0)
        self.optimizer.zero_grad()
        error = Ei_label - Ei_predict
        #TODO: as the comment from RuNNer, the error why scaler by atom_num in Etot is because of
        #Etot is the sum of Ei, so maybe we don't need to scaler the egroup error. NEED CHECK!
        error = error 
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        # if self.is_distributed:
        #     if self.distributed_backend == "horovod":
        #         import horovod as hvd

        #         error = hvd.torch.allreduce(error)
        #     elif self.distributed_backend == "torch":
        #         dist.all_reduce(error)
        #         error /= dist.get_world_size()
        
        _Ei_predict = update_prefactor * Ei_predict
        _Ei_predict[mask] = -1.0 * _Ei_predict[mask]

        _Ei_predict.sum().backward()
        error = error * math.sqrt(bs)
        self.optimizer.step(error)
        return Ei_predict
    
        # # |<---group1--->|<---group2--->|<-- ... -->|<---groupN--->| 
        # #     randomly choose atomidx for udpate 
        # selectNum = int(natoms_sum * 0.2) # choose 20%
        # updateNum = 5
        # selectNum = int(selectNum / updateNum ) * updateNum
        # weights = torch.ones(natoms_sum)  
        # selectedIdx = torch.multinomial(weights, selectNum, replacement=False)      # randomly choose atomidx for udpate
        
        # for i in range(int(selectNum / updateNum) + 1):
        #     self.optimizer.zero_grad()
        #     if train_type == "DP":
        #         Etot_predict, Ei_predict, _, _, _ = self.model(
        #             inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], is_calc_f=False
        #         )
        #     elif train_type == "NN":  # nn training
        #         Etot_predict, Ei_predict, _, _, _ = self.model(
        #             inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
        #         )
        #     else:
        #         raise Exception("Error! the train type {} is not realized!".format(train_type))     
        #     select_indexs = selectedIdx[i * updateNum : (i+1) * updateNum]     
        #     error_tmp = Ei_label[:, select_indexs] - Ei_predict[:, select_indexs]
        #     error_tmp = update_prefactor * error_tmp
        #     mask = error_tmp < 0
        #     error_tmp[mask] = -1 * error_tmp[mask]
        #     error = error_tmp.mean()

        #     if self.is_distributed:
        #         if self.distributed_backend == "horovod":
        #             import horovod as hvd

        #             error = hvd.torch.allreduce(error)
        #         elif self.distributed_backend == "torch":
        #             dist.all_reduce(error)
        #             error /= dist.get_world_size()

        #     mask = Ei_predict[:, select_indexs] < 0
        #     tmp_ei_predict = Ei_predict[:, select_indexs] * update_prefactor
        #     tmp_ei_predict[mask] = -1.0 * tmp_ei_predict[mask]

        #     # In order to solve a pytorch bug, reference: https://github.com/pytorch/pytorch/issues/43259
        #     (tmp_ei_predict.sum() + Etot_predict.sum() * 0).backward()
        #     error = error * math.sqrt(bs)
        #     #print("force steping")
        #     self.optimizer.step(error)
        # return Ei_predict
    
    def __sample(
        self, atoms_selected: int, atoms_per_group: int, natoms: int
    ) -> np.ndarray:
        # natoms can be smaller than n_select !
        # dbg : fix chosen atoms 
        # np.random.seed(0)
        if atoms_selected % atoms_per_group:
            raise Exception("divider")
        index = range(natoms)
        res = np.random.choice(index, atoms_selected).reshape(-1, atoms_per_group)
        return res

    # """
    # @Description :
    # calculate dp model kpu by etot
    # @Returns     :
    # @Author       :wuxingxing
    # """
    def cal_kpu_etot(self, 
                list_neigh,   # int32
                type_maps,    # int32
                atom_types,    # int32
                ImageDR,      # float64
                nghost: int
                ) -> None:
        Etot_predict, _, _, _, _ = self.model(list_neigh, type_maps, atom_types, ImageDR, nghost, is_calc_f=False)
        natoms_sum = list_neigh.shape[1]
        self.optimizer.set_grad_prefactor(natoms_sum)
        self.optimizer.zero_grad()
        (Etot_predict / natoms_sum).backward()
        etot_kpu = self.optimizer.cal_kpu()
        # self.optimizer.step(None)
        return etot_kpu, Etot_predict

    # """
    # @Description :
    # calculate kpu by force
    # 1. random select 50% atoms
    # 2. force_x, force_y, force_z of each atom do backward() then calculat its kpu
    # 3. force kpu = mean of these kpus
    # @Returns     :
    # @Author       :wuxingxing
    # """
    def cal_kpu_force(self, 
                list_neigh,   # int32
                type_maps,    # int32
                atom_types,    # int32
                ImageDR,      # float64
                nghost: int
                ) -> None:
        """
        randomly generate n different nums of int type in the range of [start, end)
        """
        def get_random_nums(start, end, n):
            random.seed(2024)
            numsArray = set()
            while len(numsArray) < n:
                numsArray.add(random.randint(start, end-1))
            return list(numsArray)
        natoms_sum = list_neigh.shape[1]
        self.optimizer.set_grad_prefactor(1) #natoms_sum * self.atoms_per_group * 3
        natom_list = get_random_nums(0, natoms_sum, int(0.5*natoms_sum))

        # column_name=["atom_index", "kpu_x", "kpu_y", "kpu_z"]
        force_kpu = []
        for i in natom_list:
            force_x_y_z_kpu = []
            force_x_y_z_kpu.append(i)
            # this j could optimized by randomly select one from 3 directions.
            for j in range(3):
                self.optimizer.zero_grad()
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(list_neigh, type_maps, atom_types, ImageDR, nghost, is_calc_f=True)
                #xyz
                (Force_predict[0][i][j] + Force_predict.sum() * 0 + Etot_predict.sum() * 0).backward()
                # Force_predict[0][i][j].backward()
                f_kpu = self.optimizer.cal_kpu()
                force_x_y_z_kpu.append(float(f_kpu))
                # self.optimizer.step(None)
            force_kpu.append(force_x_y_z_kpu)
        
        return np.array(force_kpu), Force_predict
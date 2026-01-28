from logging import exception
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import numpy as np
import math
#import parameters as pm
import default_para as pm

class GKalmanFilter(nn.Module):

    def __init__(self, model, kalman_lambda, kalman_nue, device):
        super(GKalmanFilter, self).__init__()
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.device = device
        self.model = model

        self.n_select = 24
        self.Force_group_num = 6
        self.n_select_eg = 12
        self.Force_group_num_eg = 3
        self.__init_P()

    def __init_P(self):
        param_num = 0
        self.weights_index = [param_num]

        for name, param in self.model.named_parameters():
            param_num += param.data.nelement()
            self.weights_index.append(param_num)
        
        #print (self.weights_index)
        self.P = torch.eye(param_num).to(self.device)

    def __update(self, H, error, weights):

        """
            1. get the Kalman Gain Matrix
            
        """ 
        A = 1 / (self.kalman_lambda + torch.matmul(torch.matmul(H.T, self.P), H))
        
        K = torch.matmul(self.P, H)
        K = torch.matmul(K, A)

        """
            2. update weights
            put back the new weights
        """
        #print (weights[self.weights_index[-1]-40:])

        weights += K * error

        i = 0

        for name, param in self.model.named_parameters():

            start_index = self.weights_index[i]

            end_index = self.weights_index[i + 1]

            param.data = weights[start_index:end_index].reshape(param.data.T.shape).T
            i += 1 

        """
        3. update P
        """

        self.P = (1 / self.kalman_lambda) * (
            self.P - torch.matmul(torch.matmul(K, H.T), self.P)
        )
        self.P = (self.P + self.P.T) / 2

        """
        4. update kalman_lambda 
        """
        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        for name, param in self.model.named_parameters():
            param.grad.detach_()
            param.grad.zero_()

    def __get_random_index(self, Force_label, n_select, Force_group_num, natoms_img):
        total_atom = 0
        atom_list = [0]
        select_list = []
        random_index = [[[], []] for i in range(math.ceil(n_select / Force_group_num))]

        for i in range(pm.ntypes):
            total_atom += natoms_img[0, i + 1]
            atom_list.append(total_atom)
        random_list = list(range(total_atom))
        for i in range(n_select):
            select_list.append(random_list.pop(random.randint(0, total_atom - i - 1)))

        tmp = 0
        tmp2 = 0
        Force_shape = Force_label.shape[0] * Force_label.shape[1]

        tmp_1 = list(range(Force_label.shape[0]))
        tmp_2 = select_list

        for i in tmp_1:
            for k in tmp_2:
                random_index[tmp][0].append(i)
                random_index[tmp][1].append(k)
                tmp2 += 1
                if tmp2 % Force_group_num == 0:
                    tmp += 1
                if tmp2 == Force_shape:
                    break
                    
        return random_index

    def update_energy(self, inputs, Etot_label, update_prefactor=1):

        time_start = time.time()

        # forward proppgation 
        Etot_predict, Ei_predict, force_predict = self.model(inputs[0], 
                                                             inputs[1], 
                                                             inputs[2], 
                                                             inputs[3], 
                                                             inputs[4], 
                                                             inputs[5])

        errore = Etot_label.item() - Etot_predict.item()
        natoms_sum = inputs[3][0, 0]

        errore = errore / natoms_sum

        if errore < 0:
            errore = -update_prefactor * errore
            (-1.0 * Etot_predict).backward()
        else:
            errore = update_prefactor * errore
            Etot_predict.backward() 

        i = 0

        for name, param in self.model.named_parameters():

            if i == 0:
                H = (param.grad / natoms_sum).T.reshape(param.grad.nelement(), 1)
                weights = param.data.T.reshape(param.data.nelement(), 1)
                
            else:
                H = torch.cat(
                    (H, (param.grad / natoms_sum).T.reshape(param.grad.nelement(), 1))
                )

                weights = torch.cat(
                    (weights, param.data.T.reshape(param.data.nelement(), 1))
                ) 

            i += 1

        # update using KF
        self.__update(H, errore, weights)
        
        time_end = time.time()

        print("Global KF update Energy time:", time_end - time_start, "s")

    def update_ei(self,inputs,Ei_label, update_prefactor=1 ):
        """
            1. pick out 20% images 
            2. cut it into a several groups 
            
        """ 
        time_start = time.time()    
    
        #total number of atom in this image
        natoms_sum = inputs[3][0, 0]   

        atom_num = len(Ei_label[0])
        error = 0.0 

        # picking out several images randomly, and update with resecpt to each error

        # choose 20% 
        selectNum = int(natoms_sum * 0.2) 

        #number image per udpate. Each update perform 1 KF optimization 
        updateNum = 5

        """
        make selectNum multiple of updateNum, 
        or error will be underestmated in the last group 
            ??? more than 1 type of element? ???
        """
        selectNum = int(selectNum / updateNum ) * updateNum 

        """
            |<---group1--->|<---group2--->|<-- ... -->|<---groupN--->| 
            randomly choose atomidx for udpate 
        """
        # from numpy.random import choice 
        # selectedIdx = choice(natoms_sum.cpu().numpy(),selectNum,replace = False)    # natoms_sum is a tensor
        weights = torch.ones(natoms_sum)                                         # Weight for each atom: set 1, means equal probability
        selectedIdx = torch.multinomial(weights, selectNum, replacement=False)      # randomly choose atomidx for udpate

        # looping over chosen indices
        for i in range(int(selectNum / updateNum) + 1):
            error = 0 

            Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
            Ei_predict.requires_grad_(True)

            # averaging the error in the current group 
            for atomIdx in selectedIdx[ i * updateNum : (i+1) * updateNum]:
                error_tmp = Ei_label[0][atomIdx] - Ei_predict[0][atomIdx]

                if error_tmp < 0:
                    error_tmp = -update_prefactor * error_tmp
                    error += error_tmp
                    (-1.0*Ei_predict[0][atomIdx]).backward(retain_graph=True)

                else:
                    error += update_prefactor * error_tmp
                    Ei_predict[0][atomIdx].backward(retain_graph=True)

            # averaging the error in this group 
            error /= (updateNum * natoms_sum)

            tmp_grad = 0
            i = 0

            for name, param in self.model.named_parameters():
                if i==0:
                    tmp_grad = param.grad
                    if tmp_grad == None:
                        tmp_grad = torch.tensor([0.0])
                    H = (tmp_grad / (updateNum * natoms_sum)).T.reshape(tmp_grad.nelement(), 1)
                    weights = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    tmp_grad = param.grad
                    if tmp_grad == None:
                        tmp_grad = torch.tensor([0.0])
                    H_new = (tmp_grad / (updateNum * natoms_sum)).T.reshape(tmp_grad.nelement(), 1)
                    H = torch.cat((H,H_new))
                    weights = torch.cat((weights, param.data.T.reshape(param.data.nelement(), 1))) 

                i+=1

            self.__update(H, error,  weights )

        time_end = time.time()
        print("Global KF update Atomic Energy time:", time_end - time_start, "s")
        
        
    def update_ei_and_force(self, inputs, Ei_label, force_label,update_prefactor=1 ):
        """
            1. pick out 20% images 
            2. cut it into a several groups 
        """ 
        time_start = time.time()    

        #total number of atom in this image
        natoms_sum = inputs[3][0, 0]   

        atom_num = len(Ei_label[0])
        error = 0.0 

        # picking out several images randomly, and update with resecpt to each error

        # choose 20% 
        selectNum = int(natoms_sum * 0.2) 

        #number image per udpate. Each update perform 1 KF optimization 
        updateNum = 5

        """
        make selectNum multiple of updateNum, 
        or error will be underestmated in the last group 
            ??? more than 1 type of element? ???
        """
        selectNum = int(selectNum / updateNum ) * updateNum 

        """
            |<---group1--->|<---group2--->|<-- ... -->|<---groupN--->| 
            randomly choose atomidx for udpate 
        """
        from numpy.random import choice 
        selectedIdx = choice(natoms_sum,selectNum,replace = False)

        # looping over chosen indices
        for i in range(int(selectNum / updateNum) + 1):
            error_ei = 0 
            error_force  = 0 

            Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
            
            Ei_predict.requires_grad_(True)
            force_predict.requires_grad_(True)

            # averaging the error in the current group 
            for atomIdx in selectedIdx[ i * updateNum : (i+1) * updateNum]:
                error_tmp = Ei_label[0][atomIdx] - Ei_predict[0][atomIdx]

                if error_tmp < 0:
                    error_tmp = -update_prefactor * error_tmp
                    error_ei += error_tmp
                    (-1.0*Ei_predict[0][atomIdx]).backward(retain_graph=True)

                else:
                    error_ei += update_prefactor * error_tmp
                    Ei_predict[0][atomIdx].backward(retain_graph=True)

                # collect force error of all the selected atoms
                for j in range(3):
                    error_tmp = force_label[0][atomIdx][j] - force_predict[0][atomIdx][j] 

                    if error_tmp < 0:
                        error_force += (-update_prefactor * error_tmp)
                        (-1.0*force_predict[0][atomIdx][j]).backward(retain_graph=True)

                    else:
                        error_force += update_prefactor * error_tmp
                        force_predict[0][atomIdx][j].backward(retain_graph=True)

            error_ei /= (updateNum * natoms_sum)
            error_force /= (updateNum * natoms_sum * 3)

            # averaging ei and force error more 
            error = 0.5* error_ei + 0.5 * error_force

            tmp_grad = 0
            i = 0

            for name, param in self.model.named_parameters():
                if i==0:
                    tmp_grad = param.grad
                    if tmp_grad == None:
                        tmp_grad = torch.tensor([0.0])
                    H = (tmp_grad / (updateNum * natoms_sum)).T.reshape(tmp_grad.nelement(), 1)
                    weights = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    tmp_grad = param.grad
                    if tmp_grad == None:
                        tmp_grad = torch.tensor([0.0])
                    H_new = (tmp_grad / (updateNum * natoms_sum)).T.reshape(tmp_grad.nelement(), 1)
                    H = torch.cat((H,H_new))
                    weights = torch.cat((weights, param.data.T.reshape(param.data.nelement(), 1))) 

                i+=1

            self.__update(H, error,  weights )

        time_end = time.time()
        print("Global KF update atomic energy and force time:", time_end - time_start, "s")


    def update_force(self, inputs, Force_label, update_prefactor=1):
        time_start = time.time()
        
        """
            now we begin to group
            NOTICE! for every force, we should repeat calculate Fatom
            because every step the weight will update, and the same dfeat/dR and de/df will get different Fatom
        """ 
        
        natoms_sum = inputs[3][0, 0]
        
        random_index = self.__get_random_index(
            Force_label, self.n_select, self.Force_group_num, inputs[3]
        )
        
        for index_i in range(len(random_index)):  # 4
            error = 0

            """
                get "measurement" which is the output of NN. 
                putting self.model call here brings huge acceleration (75%)
            """
            
            Etot_predict, Ei_predict, force_predict = self.model( inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
            force_predict.requires_grad_(True)

            for index_ii in range(len(random_index[index_i][0])):  # 6
                for j in range(3):
                    # error = 0 #if we use group , it should not be init
                    """
                    Etot_predict, Ei_predict, force_predict = self.model(
                        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
                    )
                    """ 
                    error_tmp = (
                        Force_label[random_index[index_i][0][index_ii]][
                            random_index[index_i][1][index_ii]
                        ][j]
                        - force_predict[random_index[index_i][0][index_ii]][
                            random_index[index_i][1][index_ii]
                        ][j]
                    )
                    if error_tmp < 0:
                        error_tmp = -update_prefactor * error_tmp
                        error += error_tmp
                        (
                            -1.0
                            * force_predict[random_index[index_i][0][index_ii]][
                                random_index[index_i][1][index_ii]
                            ][j]
                        ).backward(retain_graph=True)
                    else:
                        error += update_prefactor * error_tmp
                        (
                            force_predict[random_index[index_i][0][index_ii]][
                                random_index[index_i][1][index_ii]
                            ][j]
                        ).backward(retain_graph=True)

            num = len(random_index[index_i][0])
            error = (error / (num * 3.0)) / natoms_sum

            tmp_grad = 0
            i = 0
            for name, param in self.model.named_parameters():
                if i == 0:
                    tmp_grad = param.grad
                    if tmp_grad == None:  # when name==bias, the grad will be None
                        tmp_grad = torch.tensor([0.0])
                    H = ((tmp_grad / (num * 3.0)) / natoms_sum).T.reshape(
                        tmp_grad.nelement(), 1
                    )
                    weights = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    tmp_grad = param.grad
                    if tmp_grad == None:  # when name==bias, the grad will be None
                        tmp_grad = torch.tensor([0.0])
                    H = torch.cat(
                        (
                            H,
                            ((tmp_grad / (num * 3.0)) / natoms_sum).T.reshape(
                                tmp_grad.nelement(), 1
                            ),
                        )
                    )
                    
                    weights = torch.cat(
                        (weights, param.data.T.reshape(param.data.nelement(), 1))
                    )  # !!!!waring, should use T
                i += 1

            self.__update(H, error,  weights)

        time_end = time.time()
        #print("Global KF update Force time:", time_end - time_start, "s")

    def update_egroup(self, inputs, Egroup_label, update_prefactor=0.1):

        random_index = self.__get_random_index(
            Egroup_label, self.n_select_eg, self.Force_group_num_eg, inputs[3]
        )

        for index_i in range(len(random_index)):
            error = 0

            Etot_predict, Ei_predict, force_predict = self.model( inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]) 

            Egroup_predict = self.model.get_egroup(Ei_predict, inputs[4], inputs[5])  # egroup_weights, divider

            for index_ii in range(len(random_index[index_i][0])):

                error_tmp = update_prefactor * (
                    Egroup_label[random_index[index_i][0][index_ii]][
                        random_index[index_i][1][index_ii]
                    ][0]
                    - Egroup_predict[random_index[index_i][0][index_ii]][
                        random_index[index_i][1][index_ii]
                    ][0]
                )
                if error_tmp < 0:
                    error_tmp = -1.0 * error_tmp
                    error += error_tmp
                    (
                        update_prefactor
                        * (
                            -1.0
                            * Egroup_predict[random_index[index_i][0][index_ii]][
                                random_index[index_i][1][index_ii]
                            ][0]
                        )
                    ).backward(retain_graph=True)
                else:
                    error += error_tmp
                    (
                        update_prefactor
                        * (
                            Egroup_predict[random_index[index_i][0][index_ii]][
                                random_index[index_i][1][index_ii]
                            ][0]
                        )
                    ).backward(retain_graph=True)

            num = len(random_index[index_i][0])
            error = error / num

            tmp_grad = 0
            i = 0
            for name, param in self.model.named_parameters():
                if i == 0:
                    tmp_grad = param.grad
                    if tmp_grad == None:
                        tmp_grad = torch.tensor([0.0])
                    H = (tmp_grad / num).T.reshape(tmp_grad.nelement(), 1)
                    weights = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    tmp_grad = param.grad
                    if tmp_grad == None:
                        tmp_grad = torch.tensor([0.0])
                    H = torch.cat(
                        (H, (tmp_grad / num).T.reshape(tmp_grad.nelement(), 1))
                    )
                    weights = torch.cat(
                        (weights, param.data.T.reshape(param.data.nelement(), 1))
                    )
                i += 1
            self.__update(H, error, weights)


class LKalmanFilter(nn.Module):
    
    """
        only energy and force at this moment 
        Not compatible with LKF in DP 
    """
    def __init__(self, model, kalman_lambda, kalman_nue, device, nselect, groupsize, blocksize, fprefactor):
        super(LKalmanFilter, self).__init__()
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.device = device
        self.model = model
        self.n_select = nselect #24
        self.Force_group_num = groupsize #6
        self.block_size = blocksize #1024 #5120 4096
        self.f_prefactor = fprefactor
        self.__init_P() 
    
    def __init_P(self):
        self.P = []
        for name, param in self.model.named_parameters():
            param_num = param.data.nelement()
            #print(name, param_num)
            if param_num >= self.block_size:
                block_num = math.ceil(param_num / self.block_size)
                for i in range(block_num):
                    if i != block_num - 1:
                        self.P.append(torch.eye(self.block_size).to(self.device))
                    else:
                        self.P.append(
                            torch.eye(param_num - self.block_size * i).to(self.device)
                        )
            else:
                self.P.append(torch.eye(param_num).to(self.device))
        self.weights_num = len(self.P)
        
    def __split_weights(self, weight):
        param_num = weight.nelement()
        res = []
        if param_num < self.block_size:
            res.append(weight)
        else:
            block_num = math.ceil(param_num / self.block_size)
            for i in range(block_num):
                if i != block_num - 1:
                    res.append(weight[i * self.block_size : (i + 1) * self.block_size])
                else:
                    res.append(weight[i * self.block_size :])
        return res

    def __update(self, H, error, weights):
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        #time0 = time.time()
        tmp = 0
        for i in range(self.weights_num):
            tmp = tmp + (
                self.kalman_lambda + torch.matmul(torch.matmul(H[i].T, self.P[i]), H[i])
            )

        A = 1 / tmp

        #time_A = time.time()
        #print("A inversion time: ", time_A - time0)
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()

        for i in range(self.weights_num):
            time1 = time.time()
            
            # 1. get the Kalman Gain Matrix
            K = torch.matmul(self.P[i], H[i])
            K = torch.matmul(K, A)
            # torch.cuda.synchronize()
            # time2 = time.time()
            # print("update K: ", time2 - time1)

            # 2. update weights
            weights[i] = weights[i] + K * error
            # torch.cuda.synchronize()
            # time3 = time.time()
            # print("update weights time: ", time3 - time2)
            
            # 3. update P
            self.P[i] = (1 / self.kalman_lambda) * (
                self.P[i] - torch.matmul(K, torch.matmul(H[i].T, self.P[i]))
            )
            # torch.cuda.synchronize()
            # time4 = time.time()
            # print("update P: ", time4 - time3, "size ", self.P[i].shape[0])
            self.P[i] = (self.P[i] + self.P[i].T) / 2
            
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        time_end = time.time()
        #print("update all weights: ", time_end - time_A, 's')
        
        # 4. update kalman_lambda
        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        i = 0
        for name, param in self.model.named_parameters():
            param_num = param.nelement()
            if param_num < self.block_size:
                if param.ndim > 1:
                    param.data = weights[i].reshape(param.data.T.shape).T
                else:
                    param.data = weights[i].reshape(param.data.shape)
                i += 1
            else:
                block_num = math.ceil(param_num / self.block_size)
                for j in range(block_num):
                    if j == 0:
                        tmp_weight = weights[i]
                    else:
                        tmp_weight = torch.concat([tmp_weight, weights[i]], dim=0)
                    i += 1
                param.data = tmp_weight.reshape(param.data.T.shape).T
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()
        if self.device == torch.device("cuda"):    
            torch.cuda.synchronize()
        time5 = time.time()
        #print("restore weight: ", time5 - time_end, 's')

    def __sample(self, select_num, group_num, natoms):
        if select_num % group_num:
            raise Exception("divider")
        index = range(natoms)
        res = np.random.choice(index, select_num).reshape(-1, group_num)
        return res

    def update_energy(self, inputs, Etot_label, update_prefactor=1):
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        time_start = time.time()
        Etot_predict, Ei_predict, force_predict = self.model(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], is_calc_f=False
        )
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        time1 = time.time()
        #print("step(1): energy forward time: ", time1 - time_start, "s")
        
        errore = Etot_label.item() - Etot_predict.item()
        natoms_sum = inputs[3][0, 0]
        errore = errore / natoms_sum

        if errore < 0:
            errore = -update_prefactor * errore
            (-update_prefactor * Etot_predict).backward()
        else:
            errore = update_prefactor * errore
            (update_prefactor * Etot_predict).backward()
            
        time2 = time.time()
        #print("step(2): energy loss backward time: ", time2 - time1, "s")

        weights = []
        H = []
        for name, param in self.model.named_parameters():
            # H.append((param.grad / natoms_sum).T.reshape(param.grad.nelement(), 1))
            if param.ndim > 1:
                tmp = (param.grad / natoms_sum).T.reshape(param.grad.nelement(), 1)
            else:
                tmp = (param.grad / natoms_sum).reshape(param.grad.nelement(), 1)
            res = self.__split_weights(tmp)
            H = H + res
            # weights.append(param.data.T.reshape(param.data.nelement(),1))
            if param.ndim > 1:
                tmp = param.data.T.reshape(param.data.nelement(), 1)
            else:
                tmp = param.data.reshape(param.data.nelement(), 1)
            res = self.__split_weights(tmp)
            weights = weights + res
        
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        
        time_reshape = time.time()
        #print("w and b distribute time:", time_reshape - time2, "s")

        self.__update(H, errore, weights)
        
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        time_end = time.time()
        
        #print("Layerwised KF update Energy time:", time_end - time_start, "s")

    def update_force(self, inputs, Force_label, update_prefactor = 1.0):

        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        
        time_start = time.time()
        natoms_sum = inputs[3][0, 0]
        index = self.__sample(self.n_select, self.Force_group_num, natoms_sum)

        for i in range(index.shape[0]):
            if self.device == torch.device("cuda"):
                torch.cuda.synchronize()
            time0 = time.time()
            Etot_predict, Ei_predict, force_predict = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
            )

            if self.device == torch.device("cuda"):
                torch.cuda.synchronize()
            
            time1 = time.time()
            #print("step(1): force forward time: ", time1 - time0, "s")
            error_tmp = Force_label[:, index[i]] - force_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean() / natoms_sum
            tmp_force_predict = force_predict[:, index[i]]
            tmp_force_predict[mask] = -update_prefactor * tmp_force_predict[mask]
            tmp_force_predict.sum().backward()
            # loss = tmp_force_predict.sum()
            # loss.backward()
            weights = []
            H = []
            time2 = time.time()
            #print("force loss backward time: ", time2 - time1, "s")

            for name, param in self.model.named_parameters():
                # H.append((param.grad / natoms_sum).T.reshape(param.grad.nelement(), 1))
                if param.ndim > 1:
                    tmp = (param.grad / (self.Force_group_num * 3.0) / natoms_sum).T.reshape(param.grad.nelement(), 1)
                else:
                    tmp = (
                        param.grad / (self.Force_group_num * 3.0) / natoms_sum
                    ).reshape(param.grad.nelement(), 1)
                res = self.__split_weights(tmp)
                H = H + res
                # weights.append(param.data.T.reshape(param.data.nelement(),1))
                if param.ndim > 1:
                    tmp = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    tmp = param.data.reshape(param.data.nelement(), 1)
                res = self.__split_weights(tmp)
                weights = weights + res
            
            if self.device == torch.device("cuda"):
                torch.cuda.synchronize()
            
            time_reshape = time.time()
            #print("w and b distribute time:", time_reshape - time2, "s")
            self.__update(H, error, weights)
            
            if self.device == torch.device("cuda"):
                torch.cuda.synchronize()
            
            time3 = time.time()
            #print("step(3): force update time:", time3 - time2, "s")
        
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        
        time_end = time.time()

        #print("Layerwised KF update Force time:", time_end - time_start, "s")
        
class SKalmanFilter(nn.Module):
    def __init__(self, model, kalman_lambda, kalman_nue, device):
        super(SKalmanFilter, self).__init__()
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.device = device
        self.model = model
        self.n_select = 24
        self.Force_group_num = 6
        self.n_select_eg = 12
        self.Force_group_num_eg = 3
        self.block_size = 2048  # 2048 3072 4096
        self.__init_P()

    def __init_P(self):
        param_sum = 0
        for name, param in self.model.named_parameters():
            param_num = param.data.nelement()
            param_sum += param_num
            print(name, param)

        self.block_num = math.ceil(param_sum / self.block_size)
        self.last_block_size = param_sum % self.block_size
        self.padding_size = self.block_size - self.last_block_size

        P = torch.eye(self.block_size).to(self.device)
        # self.P dims ---> block_num * block_size * block_size
        self.P = P.unsqueeze(dim=0).repeat(self.block_num, 1, 1)
        self.weights_num = len(self.P)

    def __get_weights_H(self, divider):
        i = 0
        for _, param in self.model.named_parameters():
            if i == 0:
                weights = param.data.reshape(-1, 1)
                H = (param.grad / divider).T.reshape(-1, 1)
            else:
                weights = torch.cat([weights, param.data.reshape(-1, 1)], dim=0)
                tmp = (param.grad / divider).T.reshape(-1, 1)
                H = torch.cat([H, tmp], dim=0)
            i += 1
        # padding weight to match P dims
        if self.last_block_size != 0:
            weights = torch.cat(
                [weights, torch.zeros(self.padding_size, 1).to(self.device)], dim=0
            )
            H = torch.cat([H, torch.zeros(self.padding_size, 1).to(self.device)], dim=0)
        return weights, H

    def __update_weights(self, weights):

        weights = weights.reshape(-1)

        i = 0
        param_index = 0
        for _, param in self.model.named_parameters():
            param_num = param.data.nelement()
            param.data = (
                weights[param_index : param_index + param_num]
                .reshape(param.data.T.shape)
                .T
            )
            i += 1
            param_index += param_num

    def __update(self, H, error, weights):

        H = H.reshape(self.block_num, -1, 1)
        weights = weights.reshape(self.block_num, -1, 1)

        tmp = self.kalman_lambda + torch.matmul(
            torch.matmul(H.transpose(1, 2), self.P), H
        )

        A = 1 / tmp.sum(dim=0)

        """
        1. get the Kalman Gain Matrix
        """
        K = torch.matmul(self.P, H)
        K = torch.matmul(K, A)

        """
        2. update weights
        """
        weights = weights + K * error

        """
        3. update P
        """
        self.P = (1 / self.kalman_lambda) * (
            self.P - torch.matmul(torch.matmul(K, H.transpose(1, 2)), self.P)
        )
        self.P = (self.P + self.P.transpose(1, 2)) / 2

        """
        4. update kalman_lambda
        """
        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        self.__update_weights(weights)

        for name, param in self.model.named_parameters():
            param.grad.detach_()
            param.grad.zero_()


    def __sample(self, select_num, group_num, natoms):
        if select_num % group_num:
            raise Exception("divider")
        index = range(natoms)
        res = np.random.choice(index, select_num).reshape(-1, group_num)
        return res

    def update_energy(self, inputs, Etot_label, update_prefactor=1):
        time_start = time.time()
        Etot_predict, Ei_predict, force_predict = self.model(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
        )
        errore = Etot_label.item() - Etot_predict.item()
        natoms_sum = inputs[3][0, 0]
        errore = update_prefactor * errore / natoms_sum
        if errore < 0:
            errore = -1.0 * errore
            (-1.0 * Etot_predict).backward()
        else:
            Etot_predict.backward()

        weights, H = self.__get_weights_H(natoms_sum)
        self.__update(H, errore, weights)
        time_end = time.time()
        print("Sliced KF update Energy time:", time_end - time_start, "s")

    def update_force(self, inputs, Force_label, update_prefactor=2):
        time_start = time.time()
        natoms_sum = inputs[3][0, 0]
        index = self.__sample(self.n_select, self.Force_group_num, natoms_sum)

        for i in range(index.shape[0]):
            Etot_predict, Ei_predict, force_predict = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
            )
            error_tmp = Force_label[:, index[i]] - force_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean() / natoms_sum
            tmp_force_predict = force_predict[:, index[i]]
            # import ipdb;ipdb.set_trace()
            tmp_force_predict[mask] = -1 * tmp_force_predict[mask]
            tmp_force_predict.sum().backward()

            num = self.Force_group_num
            error = (error / (num * 3.0)) / natoms_sum

            weights, H = self.__get_weights_H(num * 3.0 * natoms_sum)
            self.__update(H, error, weights)

        time_end = time.time()
        print("Sliced KF update Force time:", time_end - time_start, "s")


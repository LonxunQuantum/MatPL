import torch
from torch.optim.optimizer import Optimizer


class GKFOptimizer(Optimizer):
    def __init__(
        self,
        params,
        kalman_lambda=0.98,
        kalman_nue=0.9987,
        # device=torch.device("cuda"),
    ):
        super(GKFOptimizer, self).__init__(params, {"lr": 0.1})
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        # self.device = device
        self.data_type = self.param_groups[0]["params"][0].dtype
        self.device = self.param_groups[0]["params"][0].device
        self.__init_P()

    def __init_P(self):
        param_num = 0
        self.weights_index = [param_num]
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param_num += param.data.nelement()
                self.weights_index.append(param_num)
        self.P = torch.eye(param_num, dtype=self.data_type, device=self.device)

    def __update(self, H, error, weights):

        A = 1 / (self.kalman_lambda + torch.matmul(torch.matmul(H.T, self.P), H))
        K = torch.matmul(self.P, H)
        K = torch.matmul(K, A)

        self.P = (1 / self.kalman_lambda) * (
            self.P - torch.matmul(torch.matmul(K, H.T), self.P)
        )
        self.P = (self.P + self.P.T) / 2

        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        weights += K * error
        i = 0
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param.data = (
                    weights[self.weights_index[i] : self.weights_index[i + 1]].reshape(param.data.T.shape).T
                )
                i += 1

    def set_grad_prefactor(self, grad_prefactor):
        self.grad_prefactor = grad_prefactor

    def step(self, error):

        i = 0
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                if i == 0:
                    H = (param.grad / self.grad_prefactor).T.reshape(
                        param.grad.nelement(), 1
                    )  # (param.grad / natoms_sum)?
                    weights = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    if param.grad is None:
                        H = torch.cat(
                            (
                                H,
                                torch.zeros_like(param.data.T.contiguous().reshape(param.data.nelement(), 1)),
                            )
                        )
                    else:
                        H = torch.cat(
                            (
                                H,
                                (param.grad / self.grad_prefactor).T.reshape(
                                    param.grad.nelement(), 1
                                ),
                            )
                        )
                    weights = torch.cat(
                        (weights, param.data.T.reshape(param.data.nelement(), 1))
                    )  #!!!!waring, should use T
                i += 1

        self.__update(H, error, weights)

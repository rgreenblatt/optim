import math
import torch
from torch.optim.optimizer import Optimizer
import warnings

class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam\: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay factor (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, initial_step_number=0):
        if (not callable(lr)) and (not 0.0 <= lr):
            raise ValueError("Invalid learning rate: {}".format(lr))
        if (not callable(eps)) and (not 0.0 <= eps):
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if (not callable(betas[0])) and (not 0.0 <= betas[0] < 1.0):
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if (not callable(betas[1])) and (not 0.0 <= betas[1] < 1.0):
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        self.step_number=initial_step_number

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def get(self, v):
        return v(self.step_number) if callable(v) else v

    def params(self):
        for group in self.param_groups:
            for p in group['params']:
                yield p

    def step(self, grads):
        """Performs a single optimization step.
        """
        self.step_number += 1
        p_num = 0

        for group in self.param_groups:
            for p in group['params']:
                grad = grads[p_num]
                p_num += 1
                if grad is None:
                    warnings.warn("gradient is None in Adam", RuntimeWarning)
                    continue
                grad = grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, \
                                        please consider SparseAdam instead')
                amsgrad = self.get(group['amsgrad'])
                lr = self.get(group['lr'])
                wd = self.get(group['weight_decay'])
                beta1, beta2 = self.get(group['betas'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_(-wd * lr, p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

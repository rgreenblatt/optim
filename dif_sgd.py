import torch
from optim.utils import required, DifferentiableOptimizer, set_all_parameters
from collections import defaultdict
import warnings

class DifferentiableSGD(DifferentiableOptimizer):
    def __init__(self, module, sched, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, use_in_place=False):
        if lr is not required and (not callable(lr)) and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if (not callable(momentum)) and momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if (not callable(weight_decay)) and weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if (not callable(nesterov)) and nesterov and \
           (((not callable(momentum)) and momentum <= 0) or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                             weight_decay=weight_decay, nesterov=nesterov)

        self.module = module
        self.use_in_place = use_in_place
        self.step_number=0
        self.state = defaultdict(dict) 
        self.sched = sched
        self.get_param_groups()

    def _get(self, v):
        return v(self.step_number) if callable(v) else v

    def get_state(self):
        out_state = {}
        for group in self.param_groups:
            momentum = self._get(group['momentum'])
            for name, p in group['params'][1]:
                if momentum != 0 and 'momentum_buffer' not in self.state[name]:
                    self.state[name]['momentum_buffer'] = torch.zeros_like(p.data)
                out_state[name] = self.state[name]['momentum_buffer']

        return [out_state]

    def load_state(self, state):
        assert len(state) == 1
        state = state[0]
        for group in self.param_groups:
            for name, p in group['params'][1]:
                self.state[name]['momentum_buffer'] = state[name]

    def params(self):
        self.get_param_groups()

        for group in self.param_groups:
            for _, p in group['params'][1]:
                 yield p

    def step(self, grads):
        """Performs a single optimization step.
        """
        self.step_number += 1
        p_num = 0

        self.get_param_groups()
        
        for group in self.param_groups:
            weight_decay = self._get(group['weight_decay'])
            momentum = self._get(group['momentum'])
            dampening = self._get(group['dampening'])
            nesterov = self._get(group['nesterov'])
            lr = self._get(group['lr'])
            wd = self._get(group['weight_decay'])

            new_params = {}
            setter = group['params'][0]
            for name, p in group['params'][1]:
                d_p = grads[p_num]
                p_num += 1
                if d_p is None:
                    new_params[name] = p
                    warnings.warn("gradient is None in DifferentiableSGD", RuntimeWarning)
                    continue
                if momentum != 0:
                    #assumption is that names are unique
                    param_state = self.state[name]
                    if self.use_in_place:
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                    else:
                        if 'momentum_buffer' not in param_state:
                            buf = torch.zeros_like(p.data)
                            buf = buf * momentum + d_p
                        else:
                            buf = param_state['momentum_buffer']
                            buf = buf * momentum + (1 - dampening) * d_p

                    if nesterov:
                        d_p = d_p + momentum * buf
                    else:
                        d_p = buf

                    param_state['momentum_buffer'] = buf

                if self.use_in_place:
                    p.data.add_(-lr*wd, p)
                    p.data.add_(-lr, d_p)
                else:
                    new_p = p - lr * d_p - wd * lr * p
                    new_params[name] = new_p

            if not self.use_in_place: 
                group['params'] = (setter, new_params.items())
                setter(new_params, requires_grad=False)

import torch

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()
            

class ParameterSetter():
    def __init__(self, new_params, is_buffer):
        self.new_params = new_params
        self.num_found = 0
        self.is_buffer = is_buffer

    def __call__(self, prefix, module, requires_grad):
        mod_params = module._buffers if self.is_buffer else module._parameters
        for name in mod_params:
            if (prefix+name) in self.new_params:
                self.num_found+=1
                if requires_grad and not self.is_buffer:
                    if self.new_params[prefix+name].is_leaf:
                        self.new_params[prefix+name].requires_grad_(True)
                    else:
                        self.new_params[prefix+name].retain_grad()
                mod_params[name] = self.new_params[prefix+name]

def set_all_parameters(top_module, new_params, requires_grad, is_buffer):
    setter = ParameterSetter(new_params, is_buffer)
    for prefix, module in top_module.named_modules():
        pref = "" if prefix=="" else prefix+"."
        setter(pref, module, requires_grad)

    assert setter.num_found == len(new_params)

def setter_creator(module, is_buffer=False):
    def setter(params, requires_grad=True, is_buffer=is_buffer):
        set_all_parameters(module, params, requires_grad, is_buffer)
    return setter

class DifferentiableOptimizer():

    def get_param_groups(self):
        params = self.module.get_optim_parameters(self.sched)

        self.param_groups = []

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        param_groups = list(params)

        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params'][1]
        if isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but ' + \
                            'the ordering of tensors in sets will change between runs. Please use a ' + \
                            'list instead.')
        else:
            param_group['params'] = (param_group['params'][0], list(params))
        
        for param in param_group['params'][1]:
            if not isinstance(param[1], torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization " + \
                                 "parameter " + name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params'][1]))

        if not param_set.isdisjoint(set(param_group['params'][1])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def forget(self):
        self.get_param_groups()
    
        for i, group in enumerate(self.param_groups):
            for name, p in group['params'][1]:
                param_state = self.state[(i, name)]
                for key in param_state:
                    try:
                        param_state[key].detach_()
                    except AttributeError:
                        pass
                p.detach_().requires_grad_(True)

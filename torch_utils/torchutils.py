import copy
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch import nn


class ModuleManager:
    @staticmethod
    def is_parallel(module):
        """Returns True if model is of type DP or DDP"""
        return type(module) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    @classmethod
    def de_parallel(cls, module):
        """De-parallelize a model: returns single-GPU model if model is of type DP or DDP"""
        return module.module if cls.is_parallel(module) else module

    @staticmethod
    def torch_gc():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @staticmethod
    def freeze_module(module: nn.Module, allow_train=False):
        module.requires_grad_(False)
        if not allow_train:
            # module only be allowed to eval, does not change to train mode anymore
            module.eval()
            module.train = lambda self, mode=True: self

    @staticmethod
    def quantized_by_pytorch(module: nn.Module, trace_func=None, backend='fbgemm'):
        """see https://pytorch.org/docs/stable/quantization.html"""
        module.eval()
        module.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(module, inplace=True)
        if trace_func is not None:
            with torch.no_grad():
                # help to collect the running info for quantization
                trace_func(module)
        torch.quantization.convert(module, inplace=True)

    @classmethod
    def low_memory_run(cls, module: nn.Module, call_func, device, *args, **kwargs):
        """only send the module to gpu when the module need to be run,
        and the gpu will be released after running"""
        module.to(device)
        obj = call_func(*args, **kwargs)
        module.cpu()
        cls.torch_gc()
        return obj

    @staticmethod
    def assign_device_run(module: nn.Module, call_func, device, *args, force_effect_module=True, **kwargs):
        """let module run in the assigned device"""
        if force_effect_module:
            module.to(device)

        args = [obj.to(device) if isinstance(obj, torch.Tensor) else obj for obj in args]
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        return call_func(*args, **kwargs)

    @staticmethod
    def assign_dtype_run(module: nn.Module, call_func, dtype, *args, force_effect_module=True, **kwargs):
        """let module run in the assigned dtype"""
        if force_effect_module:
            module.to(dtype)

        check = lambda obj: isinstance(obj, torch.Tensor) and obj.dtype.is_floating_point == dtype.is_floating_point
        args = [obj.to(dtype) if check(obj) else obj for obj in args]
        kwargs = {k: v.to(dtype) if check(v) else v for k, v in kwargs.items()}

        return call_func(*args, **kwargs)

    @staticmethod
    def single_batch_run(module: nn.Module, call_func, *args, **kwargs):
        """let module run one after another single batch"""
        check = lambda obj: isinstance(obj, (torch.Tensor, list, tuple))
        b = None
        for obj in args:
            if check(obj):
                b = len(obj)
                break

        temp = []
        for i in range(b):
            tmp_args = [obj[i:i + 1] if check(obj) else obj for obj in args]
            tmp_kwargs = {k: obj[i:i + 1] if check(obj) else obj for k, obj in kwargs.items()}
            rets = call_func(*tmp_args, **tmp_kwargs)
            temp.append(rets)

        return torch.cat(temp)

    @staticmethod
    def checkpoint(module: nn.Module, call_func, *args, **kwargs):
        """note, if using checkpoint, it is best not to use it in the first layer of the module,
        as it usually does not contain gradients, thought can set `x.requires_grad_(True)` to pass it,
        but it does not work yet always"""
        from torch.utils.checkpoint import checkpoint

        if module.training:  # only work on train step
            # note, if having kwargs, use `use_reentrant=False`
            return checkpoint(call_func, *args, use_reentrant=False, **kwargs)
        else:
            return call_func(*args, **kwargs)

    @classmethod
    def initialize_layers(cls, module, init_gain=0.02, init_type='normal'):
        """trace each module, initialize the variables
        if module has `initialize_layers`, use `module.initialize_layers()` to initialize"""

        def cur(current_m):
            for name, m in current_m._modules.items():
                if m is None:
                    continue

                if hasattr(m, 'initialize_layers'):
                    m.initialize_layers()
                    continue

                t = type(m)

                if t is nn.BatchNorm2d:
                    # m.eps = 1e-3
                    # m.momentum = 0.03
                    m.weight.data.normal_(1.0, init_gain)
                    m.bias.data.fill_(0.)

                elif t is nn.LayerNorm:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True

                elif t in [nn.Conv2d, nn.Linear, nn.Embedding]:
                    if init_type == 'normal':
                        nn.init.normal_(m.weight, 0.0, init_gain)
                    elif init_type == 'xavier':
                        nn.init.xavier_normal_(m.weight, gain=init_gain)
                    elif init_type == 'kaiming':
                        nn.init.kaiming_normal_(m.weight, a=0)
                    elif init_type == 'orthogonal':
                        nn.init.orthogonal_(m.weight, gain=init_gain)

                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

                elif t in [nn.ConvTranspose2d]:
                    m.weight.data.copy_(cls.bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))

                if len(m._modules) != 0:
                    cur(m)

        cur(module)

    @staticmethod
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
        f = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        ch = min(in_channels, out_channels)
        weight[range(ch), range(ch), :, :] = f
        return weight

    @staticmethod
    def get_module_by_name(module, tensor_name: str):
        if "." in tensor_name:
            splits = tensor_name.split(".")
            for split in splits:
                new_module = getattr(module, split)
                if new_module is None:
                    raise ValueError(f"{module} has no attribute {split}.")
                module = new_module

        else:
            module = getattr(module, tensor_name)

        return module

    @staticmethod
    def get_module_by_key(module, key=None, include=(), exclude=(), is_last_module=False, is_return_last_module=False):
        """

        Args:
            module:
            key (str or nn.Module):
            include (List[str or nn.Module]):
            exclude (List[str or nn.Module]):
            is_last_module:
            is_return_last_module:

        Returns:
            [[finded_module, name, full_name]]

        Examples:
            >>> ModuleManager.get_module_by_key(module, key='q')
            >>> ModuleManager.get_module_by_key(module, include=('q', 'k', 'v'), exclude=('l0.q', 'l0.k', 'l0.v'))

        """

        def cur(current_m: nn.Module, prev_name=''):
            for name, m in current_m._modules.items():
                if m is None:
                    continue

                full_name = f'{prev_name}.{name}'[1:]

                if is_last_module:
                    if is_find(full_name, m):
                        r.append((return_module(current_m, name), name, full_name))

                elif len(m._modules) == 0:
                    if is_find(full_name, m):
                        r.append((return_module(current_m, name), name, full_name))

                if len(m._modules) > 0:
                    cur(m, f'{prev_name}.{name}')

        def return_module(m, name=None):
            if is_return_last_module:
                return getattr(m, name)
            else:
                return m

        def is_find(name, m):
            flag = False
            for k in include:
                if is_last_module:
                    if (isinstance(k, str) and name.endswith(k)) or (not isinstance(k, str) and isinstance(m, k)):
                        flag = True

                else:
                    if (isinstance(k, str) and k in name) or (not isinstance(k, str) and isinstance(m, k)):
                        flag = True

            for k in exclude:
                if (isinstance(k, str) and k in name) or (not isinstance(k, str) and isinstance(m, k)):
                    flag = False

            return flag

        r = []
        if key is not None:
            include += (key,)
        cur(module)
        return r

    @classmethod
    def apply(cls, module, func, key=None, include=(), exclude=(), is_last_module=False, **func_kwargs):
        """
        Examples:
            # freeze encoder and decoder layer, train the head layer
            >>> ModuleManager.apply(nn.Module(), ModuleManager.freeze_module, include=('encoder', 'decoder'), exclude=('head', ), is_last_module=True)
        """
        objs = cls.get_module_by_key(module, key=key, include=include, exclude=exclude, is_last_module=is_last_module, is_return_last_module=True)

        for current_m, name, full_name in objs:
            func(current_m, **func_kwargs)
import math
import re
import os
import numpy as np
import warnings
import collections
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_COLA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class ColaConfig(PeftConfig):
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Cola."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    cola_alpha: float = field(default=1.0, metadata={"help": "Cola alpha"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module apart from ColA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.COLA


class ColaModel(torch.nn.Module):
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Cola with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = getattr(self.model, "config", {"model_type": "custom"})
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()

            config = self._prepare_cola_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        mark_only_cola_as_trainable(self.model)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _check_target_module_exists(self, cola_config, key):
        if isinstance(cola_config.target_modules, str):
            target_module_found = re.fullmatch(cola_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in cola_config.target_modules)
            is_using_layer_indexes = getattr(cola_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(cola_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(cola_config.layers_to_transform, int):
                            target_module_found = layer_index == cola_config.layers_to_transform
                        else:
                            target_module_found = layer_index in cola_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_new_module(self, cola_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "cola_alpha": cola_config.cola_alpha,
            "fan_in_fan_out": cola_config.fan_in_fan_out,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = cola_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = cola_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self, adapter_name):
        cola_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        cola_base = None
        for key in key_list:
            if not self._check_target_module_exists(cola_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            cola_base_i = cola_base[key] if cola_base else None

            if isinstance(target, ColaLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer(
                    adapter_name,
                    cola_config.cola_alpha,
                    cola_base_i,
                )
            elif isinstance(target, ColaLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer(
                    adapter_name,
                    cola_config.cola_alpha,
                    cola_base_i,
                )

            elif isinstance(target, ColaLayer):
                target.update_layer(
                    adapter_name,
                    cola_config.cola_alpha,
                    cola_base_i,
                )
            else:
                new_module = self._create_new_module(cola_config, adapter_name, target)
                new_module.update_layer(
                    adapter_name,
                    cola_config.cola_alpha,
                    cola_base_i,
                )
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {cola_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "cola_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                module.disable_adapters = False if enabled else True
            elif isinstance(module, ModulesToSaveWrapper):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def _get_active_adapter(self) -> str:
        active_adapter = None
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                active_adapter = module.active_adapter

        if active_adapter is None:
            raise ValueError(
                "Something went wrong, no active adapter could be found, please report the issue on GitHub"
            )
        return active_adapter

    def disable_adapter_layers(self):
        active_adapter = self._get_active_adapter()
        val = self.peft_config[active_adapter].bias
        if val != "none":
            msg = (
                f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                "output as the the base model would without adaption."
            )
            warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        """
        This method merges the ColA layers into the base model.
        """
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the ColA layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                module.unmerge()

    @staticmethod
    def _prepare_cola_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_COLA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_COLA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def _unload_and_optionally_merge(self, delta_weight, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge ColA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "cola" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, ColaLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge(delta_weight[key])
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    #
    # def add_weighted_adapter(self, adapters, weights, adapter_name, combination_type="svd"):
    #     """
    #     This method adds a new adapter by merging the given adapters with the given weights.
    #
    #     Args:
    #         adapters (list): List of adapter names to be merged.
    #         weights (list): List of weights for each adapter.
    #         adapter_name (str): Name of the new adapter.
    #         combination_type (str): Type of merging. Can be one of [`svd`, `linear`]
    #     """
    #     if adapter_name in list(self.peft_config.keys()):
    #         return
    #     for adapter in adapters:
    #         if adapter not in list(self.peft_config.keys()):
    #             raise ValueError(f"Adapter {adapter} does not exist")
    #
    #     # if there is only one adapter, we can only use linear merging
    #     combination_type = "linear" if len(adapters) == 1 else combination_type
    #
    #     # new rank is the max of all ranks of the adapters
    #     unique_ranks = list({self.peft_config[adapter].r for adapter in adapters})
    #     if combination_type == "linear":
    #         if len(unique_ranks) != 1:
    #             raise ValueError("All adapters must have the same r value when using `linear` combination_type")
    #         new_rank = unique_ranks[0]
    #     elif combination_type == "svd":
    #         new_rank = max(unique_ranks)
    #     else:
    #         raise ValueError(f"Invalid combination_type: {combination_type}")
    #
    #     self.peft_config[adapter_name] = replace(self.peft_config[adapters[0]], r=new_rank, lora_alpha=new_rank)
    #     self._find_and_replace(adapter_name)
    #     mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
    #     _freeze_adapter(self.model, adapter_name)
    #     key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
    #     for key in key_list:
    #         _, target, _ = _get_submodules(self.model, key)
    #         if isinstance(target, LoraLayer):
    #             if adapter_name in target.lora_A:
    #                 target_lora_A = target.lora_A[adapter_name].weight
    #                 target_lora_B = target.lora_B[adapter_name].weight
    #             elif adapter_name in target.lora_embedding_A:
    #                 target_lora_A = target.lora_embedding_A[adapter_name]
    #                 target_lora_B = target.lora_embedding_B[adapter_name]
    #
    #             target_lora_A.data = target_lora_A.data * 0.0
    #             target_lora_B.data = target_lora_B.data * 0.0
    #             if combination_type == "linear":
    #                 for adapter, weight in zip(adapters, weights):
    #                     if adapter in target.lora_A:
    #                         current_adapter_lora_A = target.lora_A[adapter].weight
    #                         current_adapter_lora_B = target.lora_B[adapter].weight
    #                     elif adapter in target.lora_embedding_A:
    #                         current_adapter_lora_A = target.lora_embedding_A[adapter]
    #                         current_adapter_lora_B = target.lora_embedding_B[adapter]
    #                     target_lora_A.data += current_adapter_lora_A.data * weight * target.scaling[adapter]
    #                     target_lora_B.data += current_adapter_lora_B.data
    #             elif combination_type == "svd":
    #                 target_lora_A.data, target_lora_B.data = self._svd_weighted_adapter(
    #                     adapters, weights, new_rank, target, target_lora_A, target_lora_B
    #                 )

    def delete_adapter(self, adapter_name):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        key_list = [key for key, _ in self.model.named_modules() if "cola" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, ColaLayer):
                for attr in [
                    "cola_alpha",
                    "scaling",
                ]:
                    if adapter_name in getattr(target, attr):
                        getattr(target, attr).pop(adapter_name)
                if target.active_adapter == adapter_name:
                    resetting_active_adapter = list(self.peft_config.keys())[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
                    )
                    target.active_adapter = resetting_active_adapter

    def merge_and_unload(self, delta_weight):
        return self._unload_and_optionally_merge(delta_weight)

    def unload(self):
        return self._unload_and_optionally_merge(None, merge=False)

    def flush(self):
        input = {}
        output_target = {}
        for name, module in self.named_modules():
            if isinstance(module, ColaLayer):
                if len(module.input) > 0:
                    input[name] = torch.cat(module.input, dim=0)
                if len(module.output_target) > 0:
                    output_target[name] = torch.cat(module.output_target, dim=0)
                module.input = []
                module.output_target = []
        return input, output_target

    def load_cola_base(self, cola_base):
        for name, module in self.named_modules():
            if isinstance(module, ColaLayer):
                if name in cola_base:
                    module.update_layer(cola_base=cola_base[name])
        return

    def load_lr(self, lr):
        for name, module in self.named_modules():
            if isinstance(module, ColaLayer):
                module.lr = lr
        return

    def input_buffer(self, flag):
        for name, module in self.named_modules():
            if isinstance(module, ColaLayer):
                module.if_input_buffer = flag
        return


def mark_only_cola_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = False
    return


class ColaLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.cola_alpha = {}
        self.cola_base = {}
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

        self.input = []
        self.output_target = []
        self.lr = 1.
        self.if_input_buffer = False

        self.hook = self.register_forward_hook(self.forward_hook)

    def update_layer(self, adapter_name="default", cola_alpha=None, cola_base=None):
        if cola_alpha is not None:
            self.cola_alpha[adapter_name] = cola_alpha
        self.cola_base[adapter_name] = {'dtype': torch.float32, 'model': cola_base}
        self.to(self.weight.device)

    def forward_hook(self, module, input, output):
        if self.training or self.if_input_buffer:
            input_ = input[0].detach().to('cpu')
            self.input.append(input_)
            output.requires_grad_(True)
            output.register_hook(self.backward_hook)
        return

    def backward_hook(self, grad):
        if self.training:
            grad_ = grad.detach().to('cpu')
            self.output_target[-1] = (self.output_target[-1] - self.lr * grad_).detach()
        return


class Linear(nn.Linear, ColaLayer):
    # Cola implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            cola_alpha: float = 1.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out),
            key: str = None,  # key of current layer
            **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ColaLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, cola_alpha)
        self.active_adapter = adapter_name

    def merge(self, delta_weight):
        if self.active_adapter:
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self.weight.data += self.get_delta_weight(delta_weight, self.active_adapter)
        self.merged = True
        return

    def unmerge(self, delta_weight=None):
        if delta_weight is None:
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.weight.data -= self.get_delta_weight(delta_weight, self.active_adapter)
        self.merged = False
        return

    def get_delta_weight(self, delta_weight, adapter):
        return (
                transpose(
                    delta_weight,
                    self.fan_in_fan_out,
                )
                * self.cola_alpha[adapter]
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.cola_base.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            if self.cola_base[self.active_adapter]['model'] is not None:
                x = x.to(self.cola_base[self.active_adapter]['dtype'])
                with torch.no_grad():
                    cola_output = self.cola_base[self.active_adapter]['model'](x) * self.cola_alpha[self.active_adapter]
                    cola_output.detach_()
                    if self.training:
                        self.output_target.append(cola_output.to('cpu'))
                result += cola_output
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, ColaLayer):
    # ColA implemented in a Embedding layer
    def __init__(
            self,
            adapter_name: str,
            num_embeddings: int,
            embedding_dim: int,
            cola_alpha: float = 1.,
            **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        ColaLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer(adapter_name, cola_alpha)
        self.active_adapter = adapter_name

    def unmerge(self, delta_weight=None):
        if delta_weight is None:
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.weight.data -= self.get_delta_weight(delta_weight, self.active_adapter)
        self.merged = False
        return

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self.weight.data += self.get_delta_weight(delta_weight, self.active_adapter)
        self.merged = True
        return

    def get_delta_weight(self, delta_weight, adapter):
        return transpose(delta_weight, True) * self.cola_alpha[adapter]

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return nn.Embedding.forward(self, x)
        elif not self.merged:
            result = nn.Embedding.forward(self, x)

            if self.cola_base[self.active_adapter]['model'] is not None:
                with torch.no_grad():
                    cola_output = self.cola_base[self.active_adapter]['model'](x) * self.cola_alpha[self.active_adapter]
                    cola_output.detach_()
                    if self.training:
                        self.output_target.append(cola_output.to('cpu'))
                result += cola_output
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, ColaLayer):
    # Lora implemented in a conv2d layer
    def __init__(
            self,
            adapter_name: str,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]],
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = 0,
            cola_alpha: float = 1.,
            **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        ColaLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer(adapter_name, cola_alpha)
        self.active_adapter = adapter_name

    def merge(self, delta_weight):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self.weight.data += self.get_delta_weight(delta_weight, self.active_adapter)
        self.merged = True
        return

    def unmerge(self, delta_weight=None):
        if delta_weight is None:
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.weight.data -= self.get_delta_weight(delta_weight, self.active_adapter)
        self.merged = False
        return

    def get_delta_weight(self, delta_weight, adapter):
        return delta_weight * self.cola_alpha[adapter]

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.cola_base.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            if self.cola_base[self.active_adapter]['model'] is not None:
                x = x.to(self.cola_base[self.active_adapter]['dtype'])
                with torch.no_grad():
                    cola_output = self.cola_base[self.active_adapter]['model'](x) * self.cola_alpha[self.active_adapter]
                    cola_output.detach_()
                    if self.training:
                        self.output_target.append(cola_output.to('cpu'))
                result += cola_output

        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, ColaLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                cola_alpha: float = 1.,
                **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            ColaLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.update_layer(adapter_name, cola_alpha)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.cola_base.keys():
                return result
            else:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()

                    if self.cola_base[self.active_adapter]['model'] is not None:
                        with torch.no_grad():
                            cola_output = self.cola_base[self.active_adapter]['model'](x).to(expected_dtype) * \
                                          self.cola_alpha[self.active_adapter]
                            cola_output.detach_()
                            if self.training:
                                self.output_target.append(cola_output.to('cpu'))
                        output = cola_output
                    else:
                        output = 0
                else:
                    if self.cola_base[self.active_adapter]['model'] is not None:
                        with torch.no_grad():
                            cola_output = self.cola_base[self.active_adapter]['model'](x) * \
                                          self.cola_alpha[self.active_adapter]
                            cola_output.detach_()
                            if self.training:
                                self.output_target.append(cola_output.to('cpu'))
                        output = cola_output
                    else:
                        output = 0
                result += output
            return result


    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, ColaLayer):
            # Lora implemented in a dense layer
            def __init__(
                    self,
                    adapter_name,
                    in_features,
                    out_features,
                    cola_alpha: float = 1.,
                    **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                ColaLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                self.update_layer(adapter_name, cola_alpha)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.cola_base.keys():
                    return result
                else:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.cola_base[self.active_adapter].weight.dtype)
                        if self.cola_base[self.active_adapter]['model'] is not None:
                            with torch.no_grad():
                                cola_output = self.cola_base[self.active_adapter]['model'](x).to(expected_dtype) * \
                                              self.cola_alpha[self.active_adapter]
                                cola_output.detach_()
                                if self.training:
                                    self.output_target.append(cola_output.to('cpu'))
                            output = cola_output
                        else:
                            output = 0
                    else:
                        if self.cola_base[self.active_adapter]['model'] is not None:
                            with torch.no_grad():
                                cola_output = self.cola_base[self.active_adapter]['model'](x) * \
                                              self.cola_alpha[self.active_adapter]
                                cola_output.detach_()
                                if self.training:
                                    self.output_target.append(cola_output.to('cpu'))
                            output = cola_output
                        else:
                            output = 0
                    result += output
                return result

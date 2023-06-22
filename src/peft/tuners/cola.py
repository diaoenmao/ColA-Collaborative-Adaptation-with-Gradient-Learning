# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
import os
import numpy as np
import warnings
import collections
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

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
    _freeze_boosting_model,
    _get_submodules,
    transpose,
    check_exists,
    makedir_exist_ok,
    load_intermediate_info,
    save,
    load,
    to_device,
    load_gradient_boosting_models
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class ColaConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`ColaModel`].

    Args:
        r (`int`): Cola attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Cola to.
        cola_alpha (`int`): The alpha parameter for Cola scaling.
        cola_dropout (`float`): The dropout probability for Cola layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Cola. Can be 'none', 'all' or 'cola_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Cola attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Cola."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    cola_alpha: int = field(default=None, metadata={"help": "Cola alpha"})
    cola_dropout: float = field(default=None, metadata={"help": "Cola dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Cola. Can be 'none', 'all' or 'cola_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_cola_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Cola layers."},
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
    get_delta_h: bool = field(default=False, metadata={"help": "Get input embedding and gradients for boosting"})
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use"
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.COLA


class ColaModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Cola) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`ColaConfig`]): The configuration of the Cola model.

    Returns:
        `torch.nn.Module`: The Cola model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, ColaConfig
        >>> from peft import ColaModel, ColaConfig

        >>> config = ColaConfig(
        ...     peft_type="LORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     cola_alpha=32,
        ...     target_modules=["q", "v"],
        ...     cola_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> cola_model = ColaModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import ColaConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = ColaConfig(
        ...     r=4, cola_alpha=16, target_modules=target_modules, cola_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> cola_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`ColaConfig`]): The configuration of the Cola model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.inputs = collections.defaultdict(list)
        self.grad_outputs = collections.defaultdict(list)
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_cola_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "ColaModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        cola_config = self.peft_config[adapter_name]
        get_delta_h = cola_config.get_delta_h
        if get_delta_h == True:
            # boosting model should be empty at this point
            mark_all_module_as_trainable(self.model, 'all')
        else:
            # inference mode
            mark_only_cola_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        cola_config = self.peft_config[adapter_name]
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Cola with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": cola_config.r,
            "cola_alpha": cola_config.cola_alpha,
            "cola_dropout": cola_config.cola_dropout,
            "fan_in_fan_out": cola_config.fan_in_fan_out,
            "init_cola_weights": cola_config.init_cola_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        is_using_layer_indexes = getattr(cola_config, "layers_to_transform", None) is not None
        layer_indexing_pattern = getattr(cola_config, "layers_pattern", None)

        if cola_config.inference_mode:
            gradient_boosting_models = load_gradient_boosting_models(cola_config)
        # print(key_list)
        for key in key_list:
            if isinstance(cola_config.target_modules, str):
                target_module_found = re.fullmatch(cola_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in cola_config.target_modules)

            layer_index = None
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

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, ColaLayer):
                    target.update_layer(
                        adapter_name,
                        cola_config.r,
                        cola_config.cola_alpha,
                        cola_config.cola_dropout,
                        cola_config.init_cola_weights,
                    )
                else:

                    kwargs = {
                        "r": cola_config.r,
                        "cola_alpha": cola_config.cola_alpha,
                        "cola_dropout": cola_config.cola_dropout,
                        "fan_in_fan_out": cola_config.fan_in_fan_out,
                        "key": key,
                        "get_delta_h": cola_config.get_delta_h,
                        "init_cola_weights": cola_config.init_cola_weights,
                    }

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
                        new_module = Linear4bit(
                            adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs
                        )
                    elif isinstance(target, torch.nn.Embedding):
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = target.num_embeddings, target.embedding_dim
                        new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
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
                        if cola_config.inference_mode:
                            gradient_boosting_model = gradient_boosting_models[key]
                            _freeze_boosting_model(gradient_boosting_model)
                            new_module.gradient_boosting_model = gradient_boosting_model

                    self._replace_module(parent, target_name, new_module, target)
                    # hook hooks when getting delta h, hook outside dict will be cleaned (question mark)
                    # if cola_config.get_delta_h == True:
                    #     self.hook_module(new_module, 'forward')
                    #     self.hook_module(new_module, 'backward')
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

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, ColaLayer):
                module.unmerge()

    @staticmethod
    def _prepare_cola_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "cola" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, ColaLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                else:
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].cola_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_cola_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "cola" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, ColaLayer):
                if adapter_name in target.cola_A:
                    target.cola_A[adapter_name].weight.data = target.cola_A[adapter_name].weight.data * 0.0
                    target.cola_B[adapter_name].weight.data = target.cola_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.cola_A:
                            continue
                        target.cola_A[adapter_name].weight.data += (
                                target.cola_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.cola_B[adapter_name].weight.data += target.cola_B[adapter].weight.data * weight

                elif adapter_name in target.cola_embedding_A:
                    target.cola_embedding_A[adapter_name].data = target.cola_embedding_A[adapter_name].data * 0.0
                    target.cola_embedding_B[adapter_name].data = target.cola_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.cola_embedding_A:
                            continue
                        target.cola_embedding_A[adapter_name].data += (
                                target.cola_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.cola_embedding_B[adapter_name].data += target.cola_embedding_B[adapter].data * weight


# Below code is based on https://github.com/microsoft/LoRA/blob/main/colalib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# mark all module as non-trainable for inference mode
def mark_all_module_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "cola_" not in n:
            p.requires_grad = True
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "cola_only":
        for m in model.modules():
            if isinstance(m, ColaLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


# had to adapt it for `cola_only` to work
def mark_only_cola_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "cola_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "cola_only":
        for m in model.modules():
            if isinstance(m, ColaLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class IntermediateInfo(Dataset):
    data_name = 'IntermediateInfo'

    def __init__(self, inputs_of_cur_key, grad_outputs_of_cur_key, transform=None):
        self.transform = transform
        self.id, self.data, self.target = self.make_data(train_data=inputs_of_cur_key,
                                                         train_target=grad_outputs_of_cur_key)

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self, train_data, train_target):
        train_id = np.arange(len(train_data)).astype(np.int64)
        return (train_id, train_data, train_target)


def create_gradient_boosting_datasets(peft_config):
    model_name = peft_config.base_model_name_or_path
    task_type = peft_config.task_type.value
    dataset_name = peft_config.dataset_name

    # Return a dictionary of gradient boosting models where the key is the name of the found layers.
    gradient_boosting_datasets = {}
    intermediate_info = load_intermediate_info(
        model_name=model_name,
        task_type=task_type,
        dataset_name=dataset_name,
    )

    key_list = [key for key, _ in intermediate_info['inputs'].items()]
    for key in key_list:
        inputs_of_cur_key = intermediate_info['inputs'][key]
        grad_outputs_of_cur_key = intermediate_info['grad_outputs'][key]
        gradient_boosting_datasets[key] = IntermediateInfo(inputs_of_cur_key, grad_outputs_of_cur_key)

    return gradient_boosting_datasets


class GradientBoosting(nn.Module):
    def __init__(self, in_features, out_features, adapter_name):
        super(GradientBoosting, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adapter_name = adapter_name
        # TODO: need to replace this with the actual implementation
        self.model = nn.Linear(in_features, out_features)

    def loss_fn(self, output, target, reduction='mean'):
        loss = F.mse_loss(output, target, reduction=reduction)
        return loss

    def f(self, x):
        # Pass the input through the model
        x = self.model(x)
        return x

    def forward(self, input):
        output = {}
        output['target'] = self.f(input['data'])
        # training mode
        if 'target' in input:
            output['loss'] = self.loss_fn(output['target'], input['target'])
        return output


def create_gradient_boosting_models(peft_config, model):
    # Return a dictionary of gradient boosting models where the key is the name of the found layers.
    adapter_name = model.active_adapter
    gradient_boosting_models = {}
    model = model.base_model.model
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        if isinstance(peft_config.target_modules, str):
            target_module_found = re.fullmatch(peft_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in peft_config.target_modules)

        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
            else:
                raise ValueError('Invalid layer type. Currently, COLA only supports linear layers.')
            gradient_boosting_models[key] = GradientBoosting(in_features, out_features, adapter_name)

    return gradient_boosting_models


class ColaLayer:
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        self.r = {}
        self.cola_alpha = {}
        self.scaling = {}
        # self.cola_dropout = nn.ModuleDict({})
        # self.cola_A = nn.ModuleDict({})
        # self.cola_B = nn.ModuleDict({})

        # For Embedding layer
        # self.cola_embedding_A = nn.ParameterDict({})
        # self.cola_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, cola_alpha, cola_dropout, init_cola_weights):
        self.r[adapter_name] = r
        self.cola_alpha[adapter_name] = cola_alpha
        # if cola_dropout > 0.0:
        #     cola_dropout_layer = nn.Dropout(p=cola_dropout)
        # else:
        #     cola_dropout_layer = nn.Identity()

        # self.cola_dropout.update(nn.ModuleDict({adapter_name: cola_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            # self.cola_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            # self.cola_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            # self.scaling[adapter_name] = cola_alpha / r
            pass
        # if init_cola_weights:
        #     self.reset_cola_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, cola_alpha, cola_dropout, init_cola_weights):
        self.r[adapter_name] = r
        self.cola_alpha[adapter_name] = cola_alpha
        if cola_dropout > 0.0:
            cola_dropout_layer = nn.Dropout(p=cola_dropout)
        else:
            cola_dropout_layer = nn.Identity()

        self.cola_dropout.update(nn.ModuleDict({adapter_name: cola_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.cola_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.cola_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = cola_alpha / r
        if init_cola_weights:
            self.reset_cola_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_cola_parameters(self, adapter_name):
        if adapter_name in self.cola_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.cola_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.cola_B[adapter_name].weight)
        if adapter_name in self.cola_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.cola_embedding_A[adapter_name])
            nn.init.normal_(self.cola_embedding_B[adapter_name])


class Linear(nn.Linear, ColaLayer):
    # Cola implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            cola_alpha: int = 1,
            cola_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out),
            key: str = None,  # key of current layer
            get_delta_h: bool = False,
            **kwargs,
    ):
        init_cola_weights = kwargs.pop("init_cola_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ColaLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.key = key
        self.get_delta_h = get_delta_h
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, cola_alpha, cola_dropout, init_cola_weights)
        self.active_adapter = adapter_name

        self.gradient_boosting_model = None
        self.inputs = []
        self.grad_outputs = []

        if self.get_delta_h:
            self.hook_module(self, 'forward')
            self.hook_module(self, 'backward')

    def merge(self):
        if self.active_adapter not in self.cola_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                    transpose(
                        self.cola_B[self.active_adapter].weight @ self.cola_A[self.active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.cola_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                    transpose(
                        self.cola_B[self.active_adapter].weight @ self.cola_A[self.active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward_hook_fn(self, module, input, output):
        self.inputs.append(to_device(input[0].detach(), 'cpu'))
        return

    def backward_hook_fn(self, module, grad_input, grad_output):
        self.grad_outputs.append(to_device(grad_output[0].detach(), 'cpu'))
        return

    def hook_module(self, module, type='forward'):
        if type == 'forward':
            module.register_forward_hook(self.forward_hook_fn)
        elif type == 'backward':
            module.register_full_backward_hook(self.backward_hook_fn)
        else:
            raise ValueError('type must be forward or backward')
        return

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        # if self.active_adapter not in self.cola_A.keys():
        #     return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        # if self.disable_adapters:
        #     if self.r[self.active_adapter] > 0 and self.merged:
        #         self.unmerge()
        #     result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        # elif self.r[self.active_adapter] > 0 and not self.merged:
        #     result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        #     x = x.to(self.cola_A[self.active_adapter].weight.dtype)

        #     result += (
        #         self.cola_B[self.active_adapter](
        #             self.cola_A[self.active_adapter](self.cola_dropout[self.active_adapter](x))
        #         )
        #         * self.scaling[self.active_adapter]
        #     )
        # else:
        #     result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.gradient_boosting_model is not None:
            # inference mode
            result += self.gradient_boosting_model({'data': x})['target']

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, ColaLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
            self,
            adapter_name: str,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            cola_alpha: int = 1,
            cola_dropout: float = 0.0,
            **kwargs,
    ):
        init_cola_weights = kwargs.pop("init_cola_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        ColaLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, cola_alpha, cola_dropout, init_cola_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                    transpose(
                        self.cola_embedding_B[self.active_adapter] @ self.cola_embedding_A[self.active_adapter], True
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                    transpose(
                        self.cola_embedding_B[self.active_adapter] @ self.cola_embedding_A[self.active_adapter], True
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                        transpose(
                            self.cola_embedding_B[self.active_adapter].weight
                            @ self.cola_embedding_A[self.active_adapter].weight,
                            True,
                        )
                        * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.cola_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.cola_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, ColaLayer):
        # Cola implemented in a dense layer
        def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                cola_alpha: int = 1,
                cola_dropout: float = 0.0,
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
            init_cola_weights = kwargs.pop("init_cola_weights", True)
            self.update_layer(adapter_name, r, cola_alpha, cola_dropout, init_cola_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.cola_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                            self.cola_B[self.active_adapter](
                                self.cola_A[self.active_adapter](self.cola_dropout[self.active_adapter](x))
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                            self.cola_B[self.active_adapter](
                                self.cola_A[self.active_adapter](self.cola_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                    )
                result += output
            return result


    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, ColaLayer):
            # Cola implemented in a dense layer
            def __init__(
                    self,
                    adapter_name,
                    in_features,
                    out_features,
                    r: int = 0,
                    cola_alpha: int = 1,
                    cola_dropout: float = 0.0,
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

                init_cola_weights = kwargs.pop("init_cola_weights", True)
                self.update_layer(adapter_name, r, cola_alpha, cola_dropout, init_cola_weights)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                result = super().forward(x)

                if self.disable_adapters or self.active_adapter not in self.cola_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.cola_A[self.active_adapter].weight.dtype)
                        output = (
                                self.cola_B[self.active_adapter](
                                    self.cola_A[self.active_adapter](self.cola_dropout[self.active_adapter](x))
                                ).to(expected_dtype)
                                * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                                self.cola_B[self.active_adapter](
                                    self.cola_A[self.active_adapter](self.cola_dropout[self.active_adapter](x))
                                )
                                * self.scaling[self.active_adapter]
                        )
                    result += output
                return result

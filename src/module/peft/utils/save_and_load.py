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
import errno
import numpy as np
import os
import re
import pickle
import torch
import collections
from .config import PeftType, PromptLearningConfig
from .other import _get_submodules
from transformers.pytorch_utils import Conv1D


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=4)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_data(save_mode, base_path, data_path, data_key, data_list, load_func, save_func):
    cur_layer_info_path = os.path.join(base_path, data_path, data_key)
    data_to_save = data_list
    if save_mode == 'append_mode':
        prev_data_list = load_func(cur_layer_info_path, mode='pickle')
        data_to_save = prev_data_list + data_to_save
    save_func(data_to_save, cur_layer_info_path, mode='pickle')
    data_list = []
    return


def save_intermediate_info(peft_config, model, save_mode='overwrite_mode'):
    model_name = peft_config.base_model_name_or_path
    task_type = peft_config.task_type.value
    dataset_name = peft_config.dataset_name
    # Create the base_path
    base_path = os.path.join(task_type, model_name, dataset_name, 'intermediate_info')

    # Create the corresponding folders
    makedir_exist_ok(base_path)
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
                if len(target.inputs) == 0 and len(target.grad_outputs) == 0:
                    raise ValueError('Nothing to save, check cola intermediate info')

                if len(target.inputs) > 0:
                    save_data(save_mode, base_path, 'inputs', key, target.inputs, load, save)
                if len(target.grad_outputs) > 0:
                    save_data(save_mode, base_path, 'grad_outputs', key, target.grad_outputs, load, save)
            else:
                raise ValueError('Invalid layer type. Currently, COLA only supports linear layers.')
    return


def load_intermediate_info(task_type, model_name, dataset_name):
    intermediate_info = collections.defaultdict(dict)
    for sub_path in ['inputs', 'grad_outputs']:
        path = os.path.join(task_type, model_name, dataset_name, 'intermediate_info', sub_path)
        # Iterate over the files in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            # Check if the current item is a file
            if os.path.isfile(file_path):
                # Load the file
                data = load(file_path, mode='pickle')
                # Store the loaded data in the data_dict using filename as the key
                intermediate_info[sub_path][filename] = data
    return intermediate_info


def save_gradient_boosting_models(peft_config, models):
    model_name = peft_config.base_model_name_or_path
    task_type = peft_config.task_type.value
    dataset_name = peft_config.dataset_name
    path = os.path.join(task_type, model_name, dataset_name, 'intermediate_info', 'gradient_boosting_models')
    save(models, path, mode='pickle')
    return

def load_gradient_boosting_models(peft_config):
    model_name = peft_config.base_model_name_or_path
    task_type = peft_config.task_type.value
    dataset_name = peft_config.dataset_name

    gradient_boosting_models = collections.defaultdict(list)
    # path = os.path.join(task_type, model_name, dataset_name, 'intermediate_info', 'gradient_boosting_models')
    path = os.path.join(task_type, model_name, dataset_name, 'intermediate_info')
    if len(os.listdir(path)) == 0:
        raise ValueError('No gradient boosting model found')
    # Iterate over the files in the directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Load the file
            gradient_boosting_models = load(file_path, mode='pickle')
            break
    return gradient_boosting_models


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif isinstance(config, PromptLearningConfig) or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return load_result

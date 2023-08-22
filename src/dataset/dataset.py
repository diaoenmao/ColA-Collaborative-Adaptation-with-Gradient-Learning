import dataset
import numpy as np
import os
import torch
from functools import partial
from collections import defaultdict
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import default_data_collator
from config import cfg
from module import to_device

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['fpb']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        dataset_ = dataset_['train'].train_test_split(test_size=0.1, seed=cfg['seed'])
        classes = dataset_['train'].features['label'].names
        dataset_ = dataset_.map(
            lambda x: {"text_label": [classes[label] for label in x["label"]]},
            batched=True,
            num_proc=1,
        )
    elif data_name in ['raft']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        classes = [k.replace("_", " ") for k in dataset_["train"].features["Label"].names]
        dataset_ = dataset_.map(
            lambda x: {"text_label": [classes[label] for label in x["Label"]]},
            batched=True,
            num_proc=1,
        )
    elif data_name in ['glue']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        dataset_['test'] = dataset_['validation']
        del dataset_['validation']
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(batch):
    return {key: [b[key] for b in batch] for key in batch[0]}


def pad_collate(batch, tokenizer):
    return tokenizer.pad(batch, padding="longest", return_tensors="pt")


def make_data_collate(collate_mode, tokenizer=None):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    elif collate_mode == 'transformer':
        return default_data_collator
    elif collate_mode == 'pad':
        return partial(pad_collate, tokenizer=tokenizer)
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, tokenizer, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    cfg['num_steps'] = {}
    for k in dataset:
        batch_size_ = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        shuffle_ = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, shuffle=shuffle_,
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=make_data_collate(cfg['collate_mode'], tokenizer),
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, sampler=sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=make_data_collate(cfg['collate_mode'], tokenizer),
                                        worker_init_fn=np.random.seed(cfg['seed']))
        cfg['num_steps'][k] = len(data_loader[k])
    return data_loader


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


def process_dataset(dataset, tokenizer):
    text_column = cfg['text_column']
    label_column = cfg['label_column']
    if cfg['data_name'] == 'fpb':
        max_length = cfg[cfg['model_name']]['max_length']

        def preprocess_function(examples):
            inputs = examples[text_column]
            targets = examples[label_column]
            model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True,
                                     return_tensors="pt")
            labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
            return model_inputs

        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    elif cfg['data_name'] == 'raft':
        max_length = cfg[cfg['model_name']]['max_length']

        def preprocess_function(examples):
            batch_size = len(examples[text_column[0]])
            inputs = [(f"{' '.join([f'{col}: {examples[col][i]}' for col in text_column])} "
                       f"Label: ") for i in range(batch_size)]
            targets = [str(x) for x in examples[label_column]]
            model_inputs = tokenizer(inputs)
            labels = tokenizer(targets)
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                        max_length - len(sample_input_ids)) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                    "attention_mask"][i]
                labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
                model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    elif cfg['data_name'] == 'glue':
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            text_inputs = [examples[k] for k in cfg['text_column']]
            model_inputs = tokenizer(*text_inputs, truncation=True, max_length=None)
            model_inputs["labels"] = examples["label"]
            return model_inputs

        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    else:
        raise ValueError('Not valid data name')
    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    cfg['target_size'] = len(tokenizer)
    return processed_dataset

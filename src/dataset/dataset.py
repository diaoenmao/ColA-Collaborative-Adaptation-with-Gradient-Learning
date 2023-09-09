import dataset
import numpy as np
import os
import copy
import torch
from functools import partial
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
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
    elif data_name in ['raft']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        classes = [k.replace("_", " ") for k in dataset_["train"].features["Label"].names]
        dataset_ = dataset_.map(
            lambda x: {"text_label": [classes[label] for label in x["Label"]]},
            batched=True,
            num_proc=1,
        )
    elif data_name in ['fpb']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        dataset_ = dataset_['train'].train_test_split(test_size=0.1, seed=cfg['seed'])
        classes = dataset_['train'].features['label'].names
        dataset_ = dataset_.map(
            lambda x: {"text_label": [classes[label] for label in x["label"]]},
            batched=True,
            num_proc=1,
        )
    elif data_name in ['wikisql']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        dataset_['test'] = concatenate_datasets([dataset_['validation'], dataset_['test']])
        del dataset_['validation']
    elif data_name in ['samsum']:
        dataset_ = load_dataset('json', data_files={
            'train': f'{root}/train.json',
            'validation': f'{root}/val.json',
            'test': f'{root}/test.json'
        })
        dataset_['test'] = concatenate_datasets([dataset_['validation'], dataset_['test']])
        del dataset_['validation']
    elif data_name in ['e2enlg']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        dataset_['test'] = concatenate_datasets([dataset_['validation'], dataset_['test']]) 
        del dataset_['validation']
    elif data_name in ['webnlg']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        dataset_['test'] = concatenate_datasets([dataset_['dev'], dataset_['test']])
        del dataset_['dev']
    elif data_name in ['dart']:
        dataset_ = load_dataset(cfg['hf_data_name'], cfg['hf_subset_name'], cache_dir=root)
        dataset_['test'] = concatenate_datasets([dataset_['validation'], dataset_['test']]) 
        del dataset_['validation']
    # WikiSQL
    # SAMSum 
    # E2E NLG Challenge
    # WebNLG
    # DART

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
    elif cfg['data_name'] == 'wikisql':
        '''
        This example was too long and was cropped:

        {
            "phase": 1,
            "question": "How would you answer a second test question?",
            "sql": {
                "agg": 0,
                "conds": {
                    "column_index": [2],
                    "condition": ["Some Entity"],
                    "operator_index": [0]
                },
                "human_readable": "SELECT Header1 FROM table WHERE Another Header = Some Entity",
                "sel": 0
            },
            "table": "{\"caption\": \"L\", \"header\": [\"Header1\", \"Header 2\", \"Another Header\"], \"id\": \"1-10015132-9\", \"name\": \"table_10015132_11\", \"page_i..."
        }
        '''
        max_length = cfg[cfg['model_name']]['max_length']

        def preprocess_function_wikisql(examples):            
            batch_size = len(examples[label_column])

            # inputs = examples[text_column]
            inputs = [(f"{' '.join([f'{col}: {examples[col][i]}' for col in text_column])}") for i in range(batch_size)]

            targets = examples[label_column]          
            targets = [targets[i]['human_readable'] for i in range(batch_size)]
            
            # Tokenizing inputs and targets
            model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

            # Replace pad token id with -100
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            
            model_inputs["labels"] = labels
            
            return model_inputs

        processed_dataset = dataset.map(
            preprocess_function_wikisql,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on wikisql dataset",
        )
    elif cfg['data_name'] == 'samsum':
        '''
        {'id': '13818513', 'summary': 'Amanda baked cookies and will bring Jerry some tomorrow.', 
        'dialogue': "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"}
        '''
        max_length = cfg[cfg['model_name']]['max_length']

        def preprocess_function_samsum(examples):            
            inputs = examples[text_column]
            targets = examples[label_column]

            # inputs = [' '.join([' '.join(triple) for triple in inputs[i]]) for i in range(batch_size)]                
            # targets = [f"Source: {targets[i]['source']} Text: {targets[i]['text']}" for i in range(batch_size)]
            
            # Tokenizing inputs and targets
            model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

            # Replace pad token id with -100
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            
            model_inputs["labels"] = labels
            
            return model_inputs

        processed_dataset = dataset.map(
            preprocess_function_samsum,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on samsum dataset",
        )
    elif cfg['data_name'] == 'e2enlg':
        '''
        {'human_reference': 'The Vaults pub near Café Adriatic has a 5 star rating.  Prices start at £30.',
        'meaning_representation': 'name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[Café Adriatic]'}
        '''
        max_length = cfg[cfg['model_name']]['max_length']

        def preprocess_function_e2enlg(examples):            
            inputs = examples[text_column]
            targets = examples[label_column]
            
            # Tokenizing inputs and targets
            model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

            # Replace pad token id with -100
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            
            model_inputs["labels"] = labels
            
            return model_inputs

        processed_dataset = dataset.map(
            preprocess_function_e2enlg,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on e2enlg dataset",
        )

    elif cfg['data_name'] == 'webnlg':
        '''
        {'2017_test_category': '',
        'category': 'Politician',
        'eid': 'Id10',
        'lex': {'comment': ['good', 'good', 'good'],
                'lid': ['Id1', 'Id2', 'Id3'],
                'text': ['World War II had Chiang Kai-shek as a commander and United States Army soldier Abner W. Sibal.',
                        'Abner W. Sibal served in the United States Army during the Second World War and during that war Chiang Kai-shek was one of the commanders.',
                        'Abner W. Sibal, served in the United States Army and fought in World War II, one of the commanders of which, was Chiang Kai-shek.']},
        'modified_triple_sets': {'mtriple_set': [['Abner_W._Sibal | battle | World_War_II',
                                                'World_War_II | commander | Chiang_Kai-shek',
                                                'Abner_W._Sibal | militaryBranch | United_States_Army']]},
        'original_triple_sets': {'otriple_set': [['Abner_W._Sibal | battles | World_War_II', 'World_War_II | commander | Chiang_Kai-shek', 'Abner_W._Sibal | branch | United_States_Army'],
                                                ['Abner_W._Sibal | militaryBranch | United_States_Army',
                                                'Abner_W._Sibal | battles | World_War_II',
                                                'World_War_II | commander | Chiang_Kai-shek']]},
        'shape': '(X (X) (X (X)))',
        'shape_type': 'mixed',
        'size': 3}
        '''
        max_length = cfg[cfg['model_name']]['max_length']

        def preprocess_function_webnlg(examples):            
            batch_size = len(examples[label_column])

            category, modified_triple_sets = examples['category'], examples['modified_triple_sets']
            temp_targets = examples[label_column]

            inputs = []
            targets = []
            for i in range(batch_size):
                comment, text = temp_targets[i]['comment'], temp_targets[i]['text']
                for j in range(len(comment)):
                    if comment[j] == 'good':
                        inputs.append(f'category: {category[i]}, mtriple_set: {modified_triple_sets[i]["mtriple_set"][0]}')
                        targets.append(f"text: {text[j]}")
            
            # Tokenizing inputs and targets
            model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

            # Replace pad token id with -100
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            
            model_inputs["labels"] = labels
            
            return model_inputs

        processed_dataset = dataset.map(
            preprocess_function_webnlg,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on DART dataset",
        )
    elif cfg['data_name'] == 'dart':
        '''
        {'annotations': {'source': ['WikiTableQuestions_mturk'],
        'text': ['First Clearing\tbased on Callicoon, New York and location at On NYS 52 1 Mi. Youngsville']},
        'subtree_was_extended': False,
        'tripleset': [['First Clearing', 'LOCATION', 'On NYS 52 1 Mi. Youngsville'],
        ['On NYS 52 1 Mi. Youngsville', 'CITY_OR_TOWN', 'Callicoon, New York']]}
        '''
        max_length = cfg[cfg['model_name']]['max_length']

        def preprocess_function_dart(examples):            
            batch_size = len(examples[label_column])

            inputs = examples[text_column]
            targets = examples[label_column]

            inputs = [' '.join([' '.join(triple) for triple in inputs[i]]) for i in range(batch_size)]                
            targets = [f"Source: {targets[i]['source']} Text: {targets[i]['text']}" for i in range(batch_size)]
            
            # Tokenizing inputs and targets
            model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

            # Replace pad token id with -100
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            
            model_inputs["labels"] = labels
            
            return model_inputs

        processed_dataset = dataset.map(
            preprocess_function_dart,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on DART dataset",
        )
        
    else:
        raise ValueError('Not valid data name')
    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    cfg['target_size'] = len(tokenizer)
    return processed_dataset

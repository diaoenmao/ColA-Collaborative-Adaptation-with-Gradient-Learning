import os
import torch
import torch.nn as nn
from config import cfg
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer


def make_hf_model(model_name):
    if 'bart' in model_name:
        cfg['model_name_or_path'] = 'facebook/{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = 'facebook/{}'.format(model_name)
    elif 'bloom' in model_name:
        cfg['model_name_or_path'] = 'bigscience/{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = 'bigscience/{}'.format(model_name)
    elif 'bart' in model_name:
        cfg['model_name_or_path'] = 'facebook/{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = 'facebook/{}'.format(model_name)
    elif 'roberta' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = '{}'.format(model_name)
    elif 'gpt' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = '{}'.format(model_name)
    elif 't5' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
        cfg['tokenizer_name_or_path'] = '{}'.format(model_name)
    else:
        raise ValueError('Not valid model name')
    cfg['cache_model_path'] = os.path.join('output', 'model', model_name)
    cfg['cache_tokenizer_path'] = os.path.join('output', 'tokenizer', model_name)
    if cfg['task_name'] == 'clm':
        model = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'])
    elif cfg['task_name'] == 's2s':
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'])
    elif cfg['task_name'] == 'sc':
        if cfg['subset_name'] in ['mnli']:
            model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name_or_path'],
                                                                       cache_dir=cfg['cache_model_path'],
                                                                       num_labels=3)  # "num_labels" is set up in model.config
        elif cfg['subset_name'] in ['stsb']:
            model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name_or_path'],
                                                                       cache_dir=cfg['cache_model_path'], num_labels=1)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name_or_path'],
                                                                       cache_dir=cfg['cache_model_path'])
    else:
        raise ValueError('Not valid task name')
    if any(k in cfg['model_name_or_path'] for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                              padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if 'gpt' in model_name:
        model.config.pad_token_id = tokenizer.pad_token_id
    cfg['pad_token_id'] = tokenizer.pad_token_id
    return model, tokenizer

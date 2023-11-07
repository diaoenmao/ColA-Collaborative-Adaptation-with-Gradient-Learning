from config import cfg


def process_control():
    cfg['model_name'] = cfg['control']['model_name']
    cfg['task_name'] = cfg['control']['task_name']
    cfg['batch_size'] = int(cfg['control']['batch_size'])
    ft_name_list = cfg['control']['ft_name'].split('-')
    cfg['ft_name'] = ft_name_list[0]
    if 'dist_mode' in cfg['control']:
        cfg['dist_mode'] = cfg['control']['dist_mode']
    else:
        cfg['dist_mode'] = 'joint'
    cfg['split_metric'] = False
    make_data_name()
    if cfg['task_name'] in ['s2s', 'sc', 'clm']:
        cfg['collate_mode'] = 'transformer'
        cfg['bart-base'] = {'max_length': 128}
        cfg['roberta-base'] = {'max_length': 128}
        cfg['gpt2'] = {'max_length': 128}
        if 'llama' in cfg['model_name']:
            cfg[cfg['model_name']] = {'max_length': 128}
    else:
        cfg['collate_mode'] = 'dict'
        data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                      'CIFAR100': [3, 32, 32]}
        target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CIFAR100': 100}
        cfg['linear'] = {}
        cfg['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
        cfg['cnn'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
        cfg['data_shape'] = data_shape[cfg['data_name']]
        cfg['target_size'] = target_size[cfg['data_name']]
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': False, 'test': False}
    if cfg['task_name'] in ['s2s', 'sc', 'clm']:
        cfg[model_name]['optimizer_name'] = 'AdamW'
        if cfg['ft_name'] == 'full':
            cfg[model_name]['lr'] = 5e-6
        else:
            cfg[model_name]['lr'] = 3e-4
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['betas'] = (0.9, 0.999)
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['nesterov'] = True
        cfg[model_name]['num_epochs'] = 40
        cfg[model_name]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[model_name]['scheduler_name'] = 'LinearAnnealingLR'
        cfg[model_name]['warmup_ratio'] = 0.05
    else:
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['lr'] = 3e-2
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['betas'] = (0.9, 0.999)
        cfg[model_name]['weight_decay'] = 0
        cfg[model_name]['nesterov'] = True
        cfg[model_name]['num_epochs'] = 400
        cfg[model_name]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
    if ft_name_list[0] == 'cola' and len(ft_name_list) > 1:
        cfg['cola'] = {}
        cfg['cola']['num_steps'] = int(ft_name_list[2])
        hidden_size = 8
        cfg['cola']['lowrank'] = {'hidden_size': hidden_size, 'dropout': 0.0}
        cfg['cola']['linear'] = {'bias': False}
        cfg['cola']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
        cfg['cola']['embedding'] = {'hidden_size': hidden_size, 'dropout': 0.0}
        cfg['cola']['model_name'] = ft_name_list[1]
        cfg['cola']['shuffle'] = {'train': True, 'test': False}
        if cfg['task_name'] in ['s2s', 'sc', 'clm']:
            cfg['cola']['optimizer_name'] = 'AdamW'
            cfg['cola']['lr'] = 3e-4
            cfg['cola']['momentum'] = 0.9
            cfg['cola']['betas'] = (0.9, 0.999)
            cfg['cola']['weight_decay'] = 5e-4
            cfg['cola']['nesterov'] = True
            cfg['cola']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
            cfg['cola']['scheduler_name'] = 'LinearAnnealingLR'
            cfg['cola']['warmup_ratio'] = 0.05
        else:
            cfg['cola']['optimizer_name'] = 'SGD'
            cfg['cola']['lr'] = 3e-2
            cfg['cola']['momentum'] = 0.9
            cfg['cola']['betas'] = (0.9, 0.999)
            cfg['cola']['weight_decay'] = 0
            cfg['cola']['nesterov'] = True
            cfg['cola']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
            cfg['cola']['scheduler_name'] = 'CosineAnnealingLR'
    return


def make_data_name():
    data_name_list = cfg['control']['data_name'].split('-')
    if len(data_name_list) == 2:
        cfg['data_name'], cfg['subset_name'] = data_name_list
    else:
        cfg['data_name'] = data_name_list[0]
        cfg['subset_name'] = 'none'
    if cfg['task_name'] in ['s2s', 'sc', 'clm']:
        data_name_dict = {
            # https://huggingface.co/datasets/financial_phrasebank
            'fpb': {'data_name': 'financial_phrasebank',
                    'subset_name_dict': {'sa': {'subset_name': 'sentences_allagree',
                                                'text_column': 'sentence',
                                                'label_column': 'text_label'}}},
            # https://huggingface.co/datasets/ptb_text_only
            'ptb': {'data_name': 'ptb_text_only',
                    'subset_name_dict': {'none': {'subset_name': None,
                                                  'text_column': 'sentence',
                                                  'label_column': None}}},
            # https://huggingface.co/datasets/wikisql
            'wikisql': {'data_name': 'wikisql',
                        'subset_name_dict': {'none': {'subset_name': None,
                                                      'text_column': ['question', 'table'],
                                                      'label_column': 'sql'}}},
            # https://huggingface.co/datasets/samsum
            # https://paperswithcode.com/dataset/samsum-corpus
            # https://arxiv.org/src/1911.12237v2/anc
            'samsum': {'data_name': 'samsum',
                       'subset_name_dict': {'none': {'subset_name': None,
                                                     'text_column': 'dialogue',
                                                     'label_column': 'summary'}}},
            # https://huggingface.co/datasets/e2e_nlg
            'e2enlg': {'data_name': 'e2e_nlg',
                       'subset_name_dict': {'none': {'subset_name': None,
                                                     'text_column': 'meaning_representation',
                                                     'label_column': 'human_reference'}}},
            # https://huggingface.co/datasets/web_nlg
            'webnlg': {'data_name': 'web_nlg',
                       'subset_name_dict': {'2017': {'subset_name': 'webnlg_challenge_2017',
                                                     'text_column': ['category', 'modified_triple_sets'],
                                                     'label_column': 'lex'}}},
            # https://huggingface.co/datasets/dart
            'dart': {'data_name': 'dart',
                     'subset_name_dict': {'none': {'subset_name': None,
                                                   'text_column': 'hardcode, complex structure',
                                                   'label_column': 'hardcode, complex structure'}}},
            # https://huggingface.co/datasets/glue
            'glue': {'data_name': 'glue',
                     'subset_name_dict': {'cola': {'subset_name': 'cola',
                                                   'text_column': ['sentence'],
                                                   'label_column': 'label'},
                                          'mnli': {'subset_name': 'mnli',
                                                   'text_column': ['premise', 'hypothesis'],
                                                   'label_column': 'label'},
                                          'mrpc': {'subset_name': 'mrpc',
                                                   'text_column': ['sentence1', 'sentence2'],
                                                   'label_column': 'label'},
                                          'qnli': {'subset_name': 'qnli',
                                                   'text_column': ['question', 'sentence'],
                                                   'label_column': 'label'},
                                          'qqp': {'subset_name': 'qqp',
                                                  'text_column': ['question1', 'question2'],
                                                  'label_column': 'label'},
                                          'rte': {'subset_name': 'rte',
                                                  'text_column': ['sentence1', 'sentence2'],
                                                  'label_column': 'label'},
                                          'sst2': {'subset_name': 'sst2',
                                                   'text_column': ['sentence'],
                                                   'label_column': 'label'},
                                          'stsb': {'subset_name': 'stsb',
                                                   'text_column': ['sentence1', 'sentence2'],
                                                   'label_column': 'label'},  # regression
                                          # datasize is small - not reported in LORA paper
                                          'wnli': {'subset_name': 'wnli',
                                                   'text_column': ['sentence1', 'sentence2'],
                                                   'label_column': 'label'}
                                          }
                     },
            # https://huggingface.co/datasets/databricks/databricks-dolly-15k
            'dolly': {'data_name': 'databricks/databricks-dolly-15k',
                      'subset_name_dict': {'15k': {'subset_name': '15k',
                                                   'text_column': ['instruction', 'context'],
                                                   'label_column': 'response'}
                                           }

                      }
        }
        cfg['hf_data_name'] = data_name_dict[cfg['data_name']]['data_name']
        cfg['hf_subset_name'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['subset_name']
        cfg['text_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['text_column']
        cfg['label_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['label_column']
    return

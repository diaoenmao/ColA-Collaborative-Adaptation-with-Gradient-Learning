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
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i']:
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
    if model_name not in cfg:
        cfg[model_name] = {}
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i']:
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
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['nesterov'] = True
        cfg[model_name]['num_epochs'] = 400
        cfg[model_name]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
    if (ft_name_list[0] == 'cola' or ft_name_list[0] == 'dreamboothcola') and len(ft_name_list) > 1:
        cfg['cola'] = {}
        cfg['cola']['num_steps'] = int(ft_name_list[2])
        if len(ft_name_list) > 3:
            cfg['cola']['merge'] = bool(int(ft_name_list[3]))
        else:
            cfg['cola']['merge'] = False
        hidden_size = 8
        cfg['cola']['lowrank'] = {'hidden_size': hidden_size, 'dropout': 0.0}
        cfg['cola']['linear'] = {'bias': False}
        cfg['cola']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
        cfg['cola']['embedding'] = {'hidden_size': hidden_size, 'dropout': 0.0}
        cfg['cola']['model_name'] = ft_name_list[1]
        cfg['cola']['shuffle'] = {'train': True, 'test': False}
        if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i']:
            cfg['cola']['optimizer_name'] = 'AdamW'
            if 'linear' in ft_name_list[1] and (
                    (cfg['task_name'] == 'sc' and cfg['subset_name'] in ['mnli', 'sst2', 'qnli', 'qqp', 'rte']) or (
                    cfg['task_name'] == 'clm' and cfg['model_name'] == 'llama-2')):
                cfg['cola']['lr'] = 5e-6
            else:
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
            cfg['cola']['weight_decay'] = 5e-4
            cfg['cola']['nesterov'] = True
            cfg['cola']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
            cfg['cola']['scheduler_name'] = 'CosineAnnealingLR'
    if cfg['task_name'] == 't2i':
        cfg[model_name]['num_epochs'] = 4
        cfg['collate_mode'] = 'dreambooth'
        cfg[model_name]['lr'] = 1e-4
        # all settings are from the peft example: https://github.com/huggingface/peft
        cfg[model_name]['prior_loss_weight'] = 1
        cfg[model_name]['resolution'] = 512
        cfg[model_name]['num_class_image'] = 200
        # The dimension used by the LoRA update matrices
        cfg[model_name]['lora_r'] = 16
        # Scaling factor
        cfg[model_name]['lora_alpha'] = 27
        cfg[model_name]['lora_dropout'] = 0
        cfg[model_name]['lora_bias'] = "none"
        
        cfg[model_name]['noise_scheduler_name'] = 'DDPM'
        cfg[model_name]['beta_start'] = 0.00085
        cfg[model_name]['beta_end'] = 0.012
        cfg[model_name]['beta_schedule'] = 'scaled_linear'
        cfg[model_name]['num_train_timesteps'] = 1000

        cfg[model_name]['scheduler_name'] = 'ConstantLR'
        cfg[model_name]['factor'] = 1
        if 'cola' in cfg:
            cfg['cola']['scheduler_name'] = 'ConstantLR'
            cfg['cola']['factor'] = 1
            cfg['cola']['lr'] = 1e-4

        cfg[model_name]['num_inference_steps'] = 50
        cfg[model_name]['guidance_scale'] = 7.5
        
    cfg['test_computation'] = False
    if cfg['test_computation']:
        cfg['num_test_iter'] = 10
        # cfg['device_cola'] = 'cpu'
        cfg['device_cola'] = cfg['device']
        cfg['offload_device'] = 'cpu'
        # cfg['offload_device'] = cfg['device']
        # cfg['offload_device'] = 'cuda:1'
        cfg['time_used'] = []
        cfg['time_used_cola'] = []
        cfg['mem_used'] = []
        cfg['mem_used_cola'] = []
    else:
        # cfg['device_cola'] = 'cpu'
        cfg['device_cola'] = cfg['device']
        # cfg['offload_device'] = 'cpu'
        cfg['offload_device'] = cfg['device']
    return


def make_data_name():
    data_name_list = cfg['control']['data_name'].split('-')
    if len(data_name_list) == 2:
        cfg['data_name'], cfg['subset_name'] = data_name_list
    else:
        cfg['data_name'] = data_name_list[0]
        cfg['subset_name'] = 'none'
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i']:
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

                  },
        # Dataset: https://github.com/google/dreambooth
        # DreamBooth paper: https://arxiv.org/pdf/2208.12242.pdf
        'dbdataset': { 'data_name': 'DreamBoothDataset',
                        'subset_name_dict': {
                            'backpack': {'subset_name': 'backpack',
                                         'class': 'backpack',
                                         'category': 'object'},
                            'backpack_dog': {'subset_name': 'backpack_dog',
                                             'class': 'backpack',
                                             'category': 'object'},
                            'bear_plushie': {'subset_name': 'bear_plushie',
                                             'class': 'stuffed animal',
                                             'category': 'toy'},
                            'berry_bowl': {'subset_name': 'berry_bowl',
                                           'class': 'bowl',
                                           'category': 'object'},
                            'can': {'subset_name': 'can', 'class': 'can', 'category': 'object'},
                            'candle': {'subset_name': 'candle', 'class': 'candle', 'category': 'object'},
                            'cat': {'subset_name': 'cat', 'class': 'cat', 'category': 'live object'},
                            'cat2': {'subset_name': 'cat2', 'class': 'cat', 'category': 'live object'},
                            'clock': {'subset_name': 'clock', 'class': 'clock', 'category': 'object'},
                            'colorful_sneaker': {'subset_name': 'colorful_sneaker',
                                                 'class': 'sneaker',
                                                 'category': 'object'},
                            'dog': {'subset_name': 'dog', 'class': 'dog', 'category': 'live object'},
                            'dog2': {'subset_name': 'dog2', 'class': 'dog', 'category': 'live object'},
                            'dog3': {'subset_name': 'dog3', 'class': 'dog', 'category': 'live object'},
                            'dog5': {'subset_name': 'dog5', 'class': 'dog', 'category': 'live object'},
                            'dog6': {'subset_name': 'dog6', 'class': 'dog', 'category': 'live object'},
                            'dog7': {'subset_name': 'dog7', 'class': 'dog', 'category': 'live object'},
                            'dog8': {'subset_name': 'dog8', 'class': 'dog', 'category': 'live object'},
                            'duck_toy': {'subset_name': 'duck_toy', 'class': 'toy', 'category': 'toy'},
                            'fancy_boot': {'subset_name': 'fancy_boot',
                                           'class': 'boot',
                                           'category': 'object'},
                            'grey_sloth_plushie': {'subset_name': 'grey_sloth_plushie',
                                                   'class': 'stuffed animal',
                                                   'category': 'toy'},
                            'monster_toy': {'subset_name': 'monster_toy',
                                            'class': 'toy',
                                            'category': 'toy'},
                            'pink_sunglasses': {'subset_name': 'pink_sunglasses',
                                                'class': 'glasses',
                                                'category': 'accessory'},
                            'poop_emoji': {'subset_name': 'poop_emoji',
                                           'class': 'toy',
                                           'category': 'toy'},
                            'rc_car': {'subset_name': 'rc_car', 'class': 'toy', 'category': 'toy'},
                            'red_cartoon': {'subset_name': 'red_cartoon',
                                            'class': 'cartoon',
                                            'category': 'object'},
                            'robot_toy': {'subset_name': 'robot_toy', 'class': 'toy', 'category': 'toy'},
                            'shiny_sneaker': {'subset_name': 'shiny_sneaker',
                                              'class': 'sneaker',
                                              'category': 'object'},
                            'teapot': {'subset_name': 'teapot', 'class': 'teapot', 'category': 'object'},
                            'vase': {'subset_name': 'vase', 'class': 'vase', 'category': 'object'},
                            'wolf_plushie': {'subset_name': 'wolf_plushie',
                                             'class': 'stuffed animal',
                                             'category': 'toy'}}
        }
    }
    if cfg['data_name'] == 'dbdataset':
        cfg['unique_id'] = 'sks'
        cfg['unique_class'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['class']
        return
    cfg['hf_data_name'] = data_name_dict[cfg['data_name']]['data_name']
    cfg['hf_subset_name'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['subset_name']
    cfg['text_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['text_column']
    cfg['label_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['label_column']
    return

from config import cfg


def process_control():
    make_data_name()
    if cfg['data_name'] in ['glue']:
        cfg['collate_mode'] = 'pad'
    else:
        cfg['collate_mode'] = 'transformer'
    cfg['bart-base'] = {'max_length': 128}
    cfg['bloomz-560m'] = {'max_length': 64}
    cfg['roberta-large'] = {}
    cfg['model_name'] = cfg['control']['model_name']
    cfg['task_name'] = cfg['control']['task_name']
    ft_name_list = cfg['control']['ft_name'].split('-')
    cfg['ft_name'] = ft_name_list[0]
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'AdamW'
    if cfg['ft_name'] == 'full':
        cfg[model_name]['lr'] = 5e-5
    else:
        cfg[model_name]['lr'] = 1e-3
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['betas'] = (0.9, 0.999)
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['num_epochs'] = 8
    cfg[model_name]['batch_size'] = {'train': 8, 'test': 32}
    cfg[model_name]['scheduler_name'] = 'LinearAnnealingLR'
    cfg[model_name]['scheduler_name'] = 'None'

    if ft_name_list[0] == 'cola' and len(ft_name_list) > 1:
        cfg['cola'] = {}
        if ft_name_list[1] == 'lr':
            cfg['cola']['model'] = {'name': ft_name_list[1], 'hidden_size': 64, 'dropout': 0.0}
        elif ft_name_list[1] == 'linear':
            cfg['cola']['model'] = {'name': ft_name_list[1]}
        elif ft_name_list[1] == 'mlp':
            cfg['cola']['model'] = {'name': ft_name_list[1], 'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2,
                                    'activation': 'relu'}
        elif ft_name_list[1] in ['skmlp']:
            cfg['cola']['model'] = {'name': ft_name_list[1]}
        else:
            raise ValueError('Not valid cola model')
        cfg['cola']['shuffle'] = {'train': True, 'test': False}
        cfg['cola']['optimizer_name'] = 'AdamW'
        cfg['cola']['lr'] = 1
        cfg['cola']['momentum'] = 0.9
        cfg['cola']['betas'] = (0.9, 0.999)
        cfg['cola']['weight_decay'] = 5e-4
        cfg['cola']['nesterov'] = True
        cfg['cola']['num_steps'] = int(ft_name_list[2])
        cfg['cola']['num_epochs'] = int(ft_name_list[3])
        cfg['cola']['batch_size'] = {'train': 8, 'test': 32}
        cfg['cola']['scheduler_name'] = 'LinearAnnealingLR'
        cfg['cola_func'] = {}
        cfg['cola_func']['optimizer_name'] = 'AdamW'
        cfg['cola_func']['lr'] = 1e-3
        cfg['cola_func']['momentum'] = 0.9
        cfg['cola_func']['betas'] = (0.9, 0.999)
        cfg['cola_func']['weight_decay'] = 5e-4
        cfg['cola_func']['nesterov'] = True
        cfg['cola_func']['scheduler_name'] = 'LinearAnnealingLR'
    return


def make_data_name():
    cfg['data_name'], cfg['subset_name'] = cfg['control']['data_name'].split('-')
    data_name_dict = {'fpb': {'data_name': 'financial_phrasebank',
                              'subset_name_dict': {'sa': {'subset_name': 'sentences_allagree',
                                                          'text_column': 'sentence',
                                                          'label_column': 'text_label'}}},
                      # https://huggingface.co/datasets/wikisql
                      'wikisql': {'data_name': 'wikisql',
                              'subset_name_dict': {'main': {'subset_name': None,
                                                          'text_column': ['question', 'table'],
                                                          'label_column': 'sql'}}},
                      # https://huggingface.co/datasets/samsum
                      # https://paperswithcode.com/dataset/samsum-corpus
                      # https://arxiv.org/src/1911.12237v2/anc
                      'samsum': {'data_name': 'samsum',
                              'subset_name_dict': {'main': {'subset_name': None,
                                                          'text_column': 'dialogue',
                                                          'label_column': 'summary'}}},
                      # https://huggingface.co/datasets/e2e_nlg
                      'e2enlg': {'data_name': 'e2e_nlg',
                              'subset_name_dict': {'main': {'subset_name': None,
                                                          'text_column': 'meaning_representation',
                                                          'label_column': 'human_reference'}}},
                      # https://huggingface.co/datasets/web_nlg
                      'webnlg': {'data_name': 'web_nlg',
                              'subset_name_dict': {'2017': {'subset_name': 'webnlg_challenge_2017',
                                                          'text_column': ['category', 'modified_triple_sets'],
                                                          'label_column': 'lex'}}},    
                      # https://huggingface.co/datasets/dart
                      'dart': {'data_name': 'dart',
                              'subset_name_dict': {'main': {'subset_name': None,
                                                          'text_column': 'hardcode, complex structure',
                                                          'label_column': 'hardcode, complex structure'}}},

                      'raft': {'data_name': 'ought/raft',
                               'subset_name_dict': {'tc': {'subset_name': 'twitter_complaints',
                                                           'text_column': ['Tweet text'],
                                                           'label_column': 'text_label'}}},

                      'glue': {'data_name': 'glue',
                               'subset_name_dict': {'cola': {'subset_name': 'cola',
                                                             'text_column': ['sentence'],
                                                             'label_column': 'label'},
                                                    'mnli': {'subset_name': 'mnli',
                                                             'text_column': ['premise', 'hypothesis'],
                                                             'label_column': 'label'},
                                                    'mnlim': {'subset_name': 'mnli_matched',
                                                              'text_column': ['premise', 'hypothesis'],
                                                              'label_column': 'label'},
                                                    'mnlimm': {'subset_name': 'mnli_mismatched',
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
                                                             'label_column': 'label'},
                                                    'wnli': {'subset_name': 'wnli',
                                                             'text_column': ['sentence1', 'sentence2'],
                                                             'label_column': 'label'}
                                                    }
                               }
                      }
    cfg['hf_data_name'] = data_name_dict[cfg['data_name']]['data_name']
    cfg['hf_subset_name'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['subset_name']
    cfg['text_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['text_column']
    cfg['label_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['label_column']
    return

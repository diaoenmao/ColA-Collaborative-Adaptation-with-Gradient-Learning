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
    cfg['ft_name'] = cfg['control']['ft_name']
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'AdamW'
    cfg[model_name]['lr'] = 1e-3
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['betas'] = (0.9, 0.999)
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['num_epochs'] = 8
    cfg[model_name]['batch_size'] = {'train': 32, 'test': 32}
    cfg[model_name]['scheduler_name'] = 'LinearAnnealingLR'
    return


def make_data_name():
    cfg['data_name'], cfg['subset_name'] = cfg['control']['data_name'].split('-')
    data_name_dict = {'fpb': {'data_name': 'financial_phrasebank',
                              'subset_name_dict': {'sa': {'subset_name': 'sentences_allagree',
                                                          'text_column': 'sentence',
                                                          'label_column': 'text_label'}}},
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

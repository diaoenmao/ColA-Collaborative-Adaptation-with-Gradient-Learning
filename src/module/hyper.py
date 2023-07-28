from config import cfg


def process_control():
    cfg['collate_mode'] = 'transformer'
    cfg['bart'] = {'max_length': 128}
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['ft_name'] = cfg['control']['ft_name']
    cfg['text_column'] = 'sentence'
    cfg['label_column'] = 'text_label'
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'AdamW'
    cfg[model_name]['lr'] = 1e-3
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['betas'] = (0.9, 0.999)
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['scheduler_name'] = 'LinearAnnealingLR'
    cfg[model_name]['num_epochs'] = 8
    cfg[model_name]['batch_size'] = {'train': 8, 'test': 16}
    return

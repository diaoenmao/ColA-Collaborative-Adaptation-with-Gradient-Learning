import argparse
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from metric import make_metric, make_logger
from model import make_model, make_optimizer, make_scheduler, make_ft_model, make_cola
from module import save, to_device, process_control, resume, makedir_exist_ok, PeftModel

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    model_path = os.path.join('output', 'model')
    model_tag_path = os.path.join(model_path, cfg['model_tag'])
    checkpoint_path = os.path.join(model_tag_path, 'checkpoint')
    best_path = os.path.join(model_tag_path, 'best')
    dataset = make_dataset(cfg['data_name'])
    model, tokenizer = make_model(cfg['model_name'])
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    result = resume(os.path.join(checkpoint_path, 'model'), resume_mode=cfg['resume_mode'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if result is None:
        cfg['epoch'] = 1
        model = make_ft_model(model)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        cola_base = make_cola(model, cfg['cola']['model_name'], cfg['dist_mode'])
        model.load_cola_base(cola_base)
        optimizer = defaultdict(list)
        scheduler = defaultdict(list)
        for k in cola_base:
            for i in range(cfg['num_split']):
                if cfg['cola']['model_name'][i] in ['lowrank', 'linear', 'mlp']:
                    cola_base[k].model[i] = cola_base[k].model[i].to(cfg['device'])
                    cola_param_k_i = cola_base[k].model[i].parameters()
                else:
                    cola_param_k_i = [torch.tensor([0.0], requires_grad=True)]
                optimizer[k].append(make_optimizer(cola_param_k_i, 'cola'))
                scheduler[k].append(make_scheduler(optimizer[k][i], 'cola'))
    else:
        cfg['epoch'] = result['epoch']
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_path, 'adapter'), is_trainable=True)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        cola_base = make_cola(model, cfg['cola']['model_name'], cfg['dist_mode'])
        for k in cola_base:
            cola_base[k].load_state_dict(result['cola_base_state_dict'][k])
        model.load_cola_base(cola_base)
        optimizer = defaultdict(list)
        scheduler = defaultdict(list)
        for k in cola_base:
            for i in range(cfg['num_split']):
                cola_base[k].model[i] = cola_base[k].model[i].to(cfg['device'])
                cola_param_k_i = cola_base[k].model[i].parameters()
                optimizer[k].append(make_optimizer(cola_param_k_i, 'cola'))
                scheduler[k].append(make_scheduler(optimizer[k][i], 'cola'))
                optimizer[k][i].load_state_dict(result['optimizer_state_dict'][k][i])
                scheduler[k][i].load_state_dict(result['scheduler_state_dict'][k][i])
        metric.load_state_dict(result['metric_state_dict'])
        logger.load_state_dict(result['logger_state_dict'])
    for epoch in range(cfg['epoch'], cfg[cfg['model_name']]['num_epochs'] + 1):
        cfg['epoch'] = epoch
        train(data_loader['train'], model, cola_base, optimizer, scheduler, metric, logger)
        test(data_loader['test'], model, cola_base, metric, logger)
        result = {'cfg': cfg, 'epoch': cfg['epoch'] + 1,
                  'cola_base_state_dict': {k: cola_base[k].state_dict() for k in cola_base},
                  'optimizer_state_dict': {k: [optimizer[k][i].state_dict() for i in
                                               range(len(optimizer[k]))] for k in optimizer},
                  'scheduler_state_dict': {k: [scheduler[k][i].state_dict() for i in
                                               range(len(scheduler[k]))] for k in scheduler},
                  'metric_state_dict': metric.state_dict(), 'logger_state_dict': logger.state_dict()}
        save(result, os.path.join(checkpoint_path, 'model'))
        model.save_pretrained(os.path.join(checkpoint_path, 'adapter'))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            makedir_exist_ok(best_path)
            shutil.copy(os.path.join(checkpoint_path, 'model'), os.path.join(best_path, 'model'))
            shutil.copytree(os.path.join(checkpoint_path, 'adapter'), os.path.join(best_path, 'adapter'),
                            dirs_exist_ok=True)
        logger.save(True)
        logger.reset()
    return


def train(data_loader, model, cola_base, optimizer, scheduler, metric, logger):
    model.train(True)
    start_time = time.time()
    input_buffer = defaultdict(list)
    output_target_buffer = defaultdict(list)
    split_buffer = defaultdict(list)
    for i, input in enumerate(data_loader):
        split_i = input['split']
        for k in cola_base:
            lr = optimizer[k][0].param_groups[0]['lr']
            cola_base[k].train(False)
            cola_base[k].make_split(split_i)
        input_size = input['labels'].size(0)
        input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                 'labels': input['labels']}
        input = to_device(input, cfg['device'])
        output = model(**input)
        input_ = {'target': input['labels']}
        output_ = {'target': output['logits'], 'loss': output['loss']}
        output['loss'] = output['loss'] * torch.prod(torch.tensor(output['logits'].shape[:-1]))
        output['loss'].backward()
        model.zero_grad()
        input_i, output_target_i = model.flush()
        for k in input_i:
            input_buffer[k].append(input_i[k])
            output_target_buffer[k].append(output_target_i[k])
            split_buffer[k].append(split_i)
        if (i + 1) % cfg['cola']['num_steps'] == 0:
            for k in input_buffer:
                input_cola = torch.cat(input_buffer[k], dim=0)
                output_target_cola = torch.cat(output_target_buffer[k], dim=0)
                split_cola = torch.cat(split_buffer[k], dim=0)
                cola_base[k].make_split(split_cola)
                input_cola = {'data': input_cola, 'target': output_target_cola}
                cola_base[k].fit(input_cola, optimizer[k], scheduler[k])
            input_buffer = defaultdict(list)
            output_target_buffer = defaultdict(list)
            split_buffer = defaultdict(list)
            evaluation = metric.evaluate('train', 'batch', input_, output_)
            logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round(
                    (cfg[cfg['model_name']]['num_epochs'] - cfg['epoch']) * batch_time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train')
            print(logger.write('train', metric.metric_name['train']))
    return


def test(data_loader, model, cola_base, metric, logger):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            for k in cola_base:
                cola_base[k].make_split(input['split'])
            input_size = input['labels'].size(0)
            input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                     'labels': input['labels']}
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['labels']}
            output_ = {'target': output['logits'], 'loss': output['loss']}
            if cfg['task_name'] == 's2s':
                output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                     max_new_tokens=cfg['max_new_tokens'])
            elif cfg['task_name'] == 'clm':
                output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                     attention_mask=input["attention_mask"],
                                                     max_new_tokens=cfg['max_new_tokens'],
                                                     eos_token_id=cfg['pad_token_id'])
            metric.add('test', input_, output_)
            evaluation = metric.evaluate('test', 'batch', input_, output_)
            logger.append(evaluation, 'test', input_size)
        evaluation = metric.evaluate('test', 'full')
        logger.append(evaluation, 'test')
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()

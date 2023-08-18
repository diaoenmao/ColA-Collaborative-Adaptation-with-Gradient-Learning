import argparse
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from itertools import chain
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, make_cola_data_loader, collate
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
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']})
    logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if result is None:
        cfg['epoch'] = 1
        model = make_ft_model(model)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        cola_base = make_cola(model) if cfg['ft_name'] == 'cola' else None
        if cfg['ft_name'] == 'cola':
            cola_param = []
            for k in cola_base:
                cola_base[k] = cola_base[k].to(cfg['device'])
                cola_param.extend(list(cola_base[k].parameters()))
            optimizer = make_optimizer(cola_param, cfg['model_name'])
        else:
            optimizer = make_optimizer(model.parameters(), cfg['model_name'])
        scheduler = make_scheduler(optimizer, cfg['model_name'])
    else:
        cfg['epoch'] = result['epoch']
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_path, 'adapter'), is_trainable=True)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        cola_base = make_cola(model) if cfg['ft_name'] == 'cola' else None
        if cfg['ft_name'] == 'cola':
            cola_param = []
            for k in cola_base:
                cola_base[k] = cola_base[k].to(cfg['device'])
                cola_param.extend(list(cola_base[k].parameters()))
            optimizer = make_optimizer(cola_param, cfg['model_name'])
        else:
            optimizer = make_optimizer(model.parameters(), cfg['model_name'])
        if cfg['ft_name'] not in ['adalora']:
            optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler = make_scheduler(optimizer, cfg['model_name'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        metric.load_state_dict(result['metric_state_dict'])
        logger.load_state_dict(result['logger_state_dict'])
    for epoch in range(cfg['epoch'], cfg[cfg['model_name']]['num_epochs'] + 1):
        cfg['epoch'] = epoch
        if cfg['ft_name'] == 'cola':
            cola_data_loader = make_cola_data_loader(dataset, tokenizer, model)
            train_cola(cola_data_loader['train'], cola_base, optimizer, scheduler, metric, logger)
            test_cola(cola_data_loader['test'], cola_base, metric, logger)
            cola_base_ = {}
            for k in cola_data_loader['train']:
                cola_base_[k] = cola_base[k]
            model.load_cola_base(cola_base_)
            logger.save(True)
            logger.reset()
        else:
            train(data_loader['train'], model, optimizer, scheduler, metric, logger)
        test(data_loader['test'], model, metric, logger)
        exit()
        result = {'cfg': cfg, 'epoch': cfg['epoch'] + 1,
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
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


def train(data_loader, model, optimizer, scheduler, metric, logger):
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input_size = input['labels'].size(0)
        input = to_device(input, cfg['device'])
        output = model(**input)
        input_ = {'target': input['labels']}
        output_ = {'target': output['logits'], 'loss': output['loss']}
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        evaluation = metric.evaluate('train', 'batch', input_, output_)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - cfg['epoch']) * batch_time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train')
            print(logger.write('train', metric.metric_name['train']))
    return


def train_cola(data_loader, model, optimizer, scheduler, metric, logger):
    for k in data_loader:
        model[k].train(True)
        start_time = time.time()
        for iter in range(10):
            for i, input in enumerate(data_loader[k]):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model[k].f(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model[k].parameters(), 1)
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                evaluation = metric.evaluate('train', 'batch', input, output, {'train': ['Loss']})
                logger.append(evaluation, 'train', n=input_size)
                if i % int((len(data_loader[k]) * cfg['log_interval']) + 1) == 0:
                    batch_time = (time.time() - start_time) / (i + 1)
                    lr = optimizer.param_groups[0]['lr']
                    epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader[k]) - i - 1)))
                    exp_finished_time = epoch_finished_time + datetime.timedelta(
                        seconds=round(
                            (cfg[cfg['model_name']]['num_epochs'] - cfg['epoch']) * batch_time * len(data_loader[k])))
                    info = {'info': ['Model: {}({})'.format(cfg['model_tag'], k),
                                     'Train Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100. * i / len(data_loader[k])),
                                     'Learning rate: {:.6f}'.format(lr),
                                     'Epoch Finished Time: {}'.format(epoch_finished_time),
                                     'Experiment Finished Time: {}'.format(exp_finished_time)]}
                    logger.append(info, 'train')
                    print(logger.write('train', ['Loss']))
    return

def test_cola(data_loader, model, metric, logger):
    with torch.no_grad():
        for k in data_loader:
            model[k].train(True)
            for i, input in enumerate(data_loader[k]):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model[k].f(input)
                evaluation = metric.evaluate('test', 'batch', input, output, {'test': ['Loss']})
                logger.append(evaluation, 'test', input_size)
            evaluation = metric.evaluate('test', 'full')
            logger.append(evaluation, 'test', input_size)
            info = {'info': ['Model: {}({})'.format(cfg['model_tag'], k), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
            logger.append(info, 'test')
            print(logger.write('test', ['Loss']))
    return

def test(data_loader, model, metric, logger):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['labels'].size(0)
            input = to_device(input, cfg['device'])
            output = model(**input)
            input_ = {'target': input['labels']}
            output_ = {'target': output['logits'], 'loss': output['loss']}
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

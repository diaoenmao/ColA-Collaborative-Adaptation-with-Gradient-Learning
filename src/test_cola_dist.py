import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_metric, make_logger
from model import make_model, make_cola
from module import save, to_device, process_control, resume, PeftModel

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
    result_path = os.path.join('output', 'result')
    model_tag_path = os.path.join(model_path, cfg['model_tag'])
    checkpoint_path = os.path.join(model_tag_path, 'checkpoint')
    best_path = os.path.join(model_tag_path, 'best')
    dataset = make_dataset(cfg['data_name'])
    model, tokenizer = make_model(cfg['model_name'])
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    cfg['split_metric'] = False
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    cfg['split_metric'] = True
    split_metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    result = resume(os.path.join(best_path, 'model'))
    model = PeftModel.from_pretrained(model, os.path.join(best_path, 'adapter'))
    model = model.to(cfg['device'])
    cola_base = make_cola(model, cfg['cola']['model_name'], cfg['dist_mode'])
    for k in cola_base:
        cola_base[k].load_state_dict(result['cola_base_state_dict'][k])
        cola_base[k] = cola_base[k].to(cfg['device'])
    model.load_cola_base(cola_base)
    cfg['epoch'] = result['epoch']
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_each_logger = make_logger(os.path.join('output', 'runs', 'test_each_{}'.format(cfg['model_tag'])))
    test_merge_logger = make_logger(os.path.join('output', 'runs', 'test_merge_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], model, cola_base, metric, test_logger)
    test_each(data_loader['test'], model, cola_base, split_metric, test_each_logger)
    if cfg['ft_name'] in ['cola']:
        delta_weight = make_delta_weight(cola_base)
        model = model.merge_and_unload(delta_weight)
        test(data_loader['test'], model, cola_base, metric, test_merge_logger)
    result = resume(os.path.join(checkpoint_path, 'model'))
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger_state_dict': {'train': result['logger_state_dict'],
                                                                       'test': test_logger.state_dict(),
                                                                       'test_each': test_each_logger.state_dict(),
                                                                       'test_merge': test_merge_logger.state_dict()}}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return


def make_delta_weight(cola_base):
    with torch.no_grad():
        delta_weight = {}
        for k in cola_base:
            delta_weight[k] = cola_base[k].make_delta_weight()
    return delta_weight


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


def test_each(data_loader, model, cola_base, metric, logger):
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
            input_target = [None for _ in range(cfg['num_split'])]
            output_target = [None for _ in range(cfg['num_split'])]
            for j in range(len(cola_base[k].indices)):
                unique_value = cola_base[k].unique_split[j]
                indices_i_j = cola_base[k].indices[j]
                input_target[unique_value] = input['labels'][indices_i_j]
                output_target[unique_value] = output['logits'][indices_i_j]
            input_ = {'target': input_target}
            output_ = {'target': output_target, 'loss': output['loss']}
            if cfg['task_name'] in ['s2s', 'clm']:
                if cfg['task_name'] == 's2s':
                    output_generate_ = model.generate(input_ids=input["input_ids"],
                                                                   max_new_tokens=cfg['max_new_tokens'])
                elif cfg['task_name'] == 'clm':
                    output_generate_ = model.generate(input_ids=input["input_ids"],
                                                                   attention_mask=input["attention_mask"],
                                                                   max_new_tokens=cfg['max_new_tokens'],
                                                                   eos_token_id=cfg['pad_token_id'])
                output_generate = [None for _ in range(cfg['num_split'])]
                for j in range(len(cola_base[k].indices)):
                    unique_value = cola_base[k].unique_split[j]
                    indices_i_j = cola_base[k].indices[j]
                    output_generate[unique_value] = output_generate_[indices_i_j]
                output_['generate'] = output_generate
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

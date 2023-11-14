import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_batchnorm_stats
from metric import make_metric, make_logger
from model import make_model
from module import save, makedir_exist_ok, to_device, process_control, resume, PeftModel

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
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    model, tokenizer = make_model(cfg['model_name'])
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    if cfg['task_name'] == 't2i':
        model_name = cfg['model_name']
        model.unet = PeftModel.from_pretrained(model.unet, os.path.join(best_path, 'adapter'))
        model = model.to(cfg['device'])
        pic_folder_path = os.path.join(result_path, cfg['model_tag'], cfg['subset_name'])
        makedir_exist_ok(pic_folder_path)
        for i in range(20):
            INSTANCE_PROMPT = f"a photo of {cfg['unique_id']} {cfg['unique_class']}"
            image = model(INSTANCE_PROMPT, num_inference_steps=cfg[model_name]['num_inference_steps'], \
                          guidance_scale=cfg[model_name]['guidance_scale']).images[0]
            image_path = os.path.join(pic_folder_path, f"{i}.png")
            image.save(image_path)
        return
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    result = resume(os.path.join(best_path, 'model'))
    model = PeftModel.from_pretrained(model, os.path.join(best_path, 'adapter'))
    model = model.to(cfg['device'])
    if cfg['model_name'] in ['cnn', 'resnet18', 'wresnet28x2']:
        model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    cfg['epoch'] = result['epoch']
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_merge_logger = make_logger(os.path.join('output', 'runs', 'test_merge_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], model, metric, test_logger)
    if cfg['ft_name'] in ['lora']:
        model = model.merge_and_unload()
        test(data_loader['test'], model, metric, test_merge_logger)
    result = resume(os.path.join(checkpoint_path, 'model'))
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger_state_dict': {'train': result['logger_state_dict'],
                                                                       'test': test_logger.state_dict(),
                                                                       'test_merge': test_merge_logger.state_dict()}}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return

def test(data_loader, model, metric, logger):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            if cfg['task_name'] in ['s2s', 'sc', 'clm']:
                input_size = input['labels'].size(0)
                input = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
                         'labels': input['labels']}
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['labels']}
                output_ = {'target': output['logits'], 'loss': output['loss']}
            else:
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(**input)
                input_ = {'target': input['target']}
                output_ = {'target': output['target'], 'loss': output['loss']}
            if cfg['task_name'] == 's2s':
                output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                     max_new_tokens=cfg['max_new_tokens'])
            elif cfg['task_name'] == 'clm':
                if cfg['data_name'] in ['dolly']:
                    output_['generate'] = model.generate(input_ids=input["input_ids"],
                                                         attention_mask=input["attention_mask"],
                                                         max_new_tokens=cfg['max_new_tokens'],
                                                         eos_token_id=cfg['pad_token_id'],
                                                         no_repeat_ngram_size=2)
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

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from metric import make_metric, make_logger
from model import make_model, freeze_model, make_cola, Router, make_delta_weight
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
    model, tokenizer = make_model(cfg['model_name'])
    output_format = 'png'
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    model_path = os.path.join('output', 'model')
    result_path = os.path.join('output', 'result')
    model_tag_path = os.path.join(model_path, cfg['model_tag'])
    best_path = os.path.join(model_tag_path, 'best')
    checkpoint_path = os.path.join(model_tag_path, 'checkpoint')
    model, tokenizer = make_model(cfg['model_name'])
    result = resume(os.path.join(checkpoint_path, 'model'))
    model.unet = PeftModel.from_pretrained(model.unet, os.path.join(checkpoint_path, 'adapter'))
    freeze_model(model.unet)
    model = model.to(cfg['device'])
    cola_base = make_cola(model.unet, cfg['cola']['model_name'])
    for k in cola_base:
        cola_base[k].load_state_dict(result['cola_base_state_dict'][k])
        cola_base[k] = cola_base[k].to(cfg['device'])
    model.unet.load_cola_base(cola_base)
    model_name = cfg['model_name']
    generate_dir = os.path.join(result_path, cfg['model_tag'], cfg['subset_name'])
    makedir_exist_ok(generate_dir)
    for i in range(30):
        INSTANCE_PROMPT = f"a photo of {cfg['unique_id']} {cfg['unique_class']}"
        image = model(INSTANCE_PROMPT, num_inference_steps=cfg[model_name]['num_inference_steps'], \
                      guidance_scale=cfg[model_name]['guidance_scale']).images[0]
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_path = os.path.join(generate_dir, f"{i}.{output_format}")
        # Save as PDF
        image.save(image_path, output_format.upper(), resolution=100.0)
    return


if __name__ == "__main__":
    main()



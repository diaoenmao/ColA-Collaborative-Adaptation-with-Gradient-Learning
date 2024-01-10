import argparse
import datetime
import numpy as np
import os
import shutil
import time
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import defaultdict
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from metric import make_metric, make_logger
from model import make_model, make_optimizer, make_scheduler, make_noise_scheduler, make_ft_model
from module import save, to_device, process_control, resume, makedir_exist_ok, PeftModel, get_available_gpus

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
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    model, tokenizer = make_model(cfg['model_name'], 'unet')
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    print('len(data_loader): ', len(data_loader['train']))
    result = resume(os.path.join(checkpoint_path, 'model'), resume_mode=cfg['resume_mode'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if result is None:
        cfg['epoch'] = 1
        model = make_ft_model(model)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        optimizer = make_optimizer(model.parameters(), cfg['model_name'])
        scheduler = make_scheduler(optimizer, cfg['model_name'])
    else:
        cfg['epoch'] = result['epoch']
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_path, 'adapter'), is_trainable=True)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        optimizer = make_optimizer(model.parameters(), cfg['model_name'])
        if cfg['ft_name'] not in ['adalora']:
            optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler = make_scheduler(optimizer, cfg['model_name'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        metric.load_state_dict(result['metric_state_dict'])
        logger.load_state_dict(result['logger_state_dict'])
    vae, _ = make_model(cfg['model_name'], 'vae')
    vae = vae.to(cfg['device'])
    text_encoder, _ = make_model(cfg['model_name'], 'text_encoder')
    text_encoder = text_encoder.to(cfg['device'])
    noise_scheduler = make_noise_scheduler(cfg['model_name'])
    for epoch in range(cfg['epoch'], cfg[cfg['model_name']]['num_epochs'] + 1):
        cfg['epoch'] = epoch
        train(data_loader['train'], model, vae, text_encoder, optimizer, scheduler, noise_scheduler, metric, logger)
        result = {'cfg': cfg, 'epoch': cfg['epoch'] + 1,
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'metric_state_dict': metric.state_dict(), 'logger_state_dict': logger.state_dict()}
        save(result, os.path.join(checkpoint_path, 'model'))
        model.save_pretrained(os.path.join(checkpoint_path, 'adapter'))
        if metric.compare(logger.mean['train/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['train/{}'.format(metric.pivot_name)])
            makedir_exist_ok(best_path)
            shutil.copy(os.path.join(checkpoint_path, 'model'), os.path.join(best_path, 'model'))
            shutil.copytree(os.path.join(checkpoint_path, 'adapter'), os.path.join(best_path, 'adapter'),
                            dirs_exist_ok=True)
        logger.reset()
    return


def train(data_loader, unet, vae, text_encoder, optimizer, scheduler, noise_scheduler, metric, logger):
    unet.train(True)
    vae.train(False)
    text_encoder.train(False)
    start_time = time.time()

    for i, input in enumerate(data_loader):
        if cfg['test_computation']:
            s = time.time()
        input = to_device(input, cfg['device'])
        with torch.no_grad():
            latents = vae.encode(input["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(input["input_ids"])[0]
        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        if cfg[cfg['model_name']]['prior_loss_weight'] > 0:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + cfg[cfg['model_name']]['prior_loss_weight'] * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        output_ = {'loss': loss}
        input_size = input['input_ids'].size(0) / 2
        optimizer.zero_grad()
        output_['loss'].backward()
        optimizer.step()
        scheduler.step()
        if cfg['test_computation']:
            cfg['time_used'].append(time.time() - s)
        evaluation = metric.evaluate('train', 'batch', None, output_)
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
            print(logger.write('train', metric.metric_name['train']), flush=True)
        if cfg['test_computation']:
            device = cfg['device']
            if cfg['device'] == 'cuda':
                gpu_ids, _ = get_available_gpus()
                device = torch.device('cuda:{}'.format(gpu_ids[-1]))
            mem_free, mem_total = torch.cuda.mem_get_info(device)
            cfg['mem_used'].append(mem_total - mem_free)
            if i == cfg['num_test_iter']:
                print(cfg['time_used'])
                print(cfg['mem_used'])
                print('Run time backward: {}({})'.format(np.mean(cfg['time_used'][1:]),
                                                         np.std(cfg['time_used'][1:])))
                print('Memory used: {}({})'.format(np.mean(cfg['mem_used'][1:]),
                                                   np.std(cfg['mem_used'][1:])))
                print('-----------------')
                exit()
    logger.save(True)
    return


if __name__ == "__main__":
    main()

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
from model import make_model, make_optimizer, make_scheduler, make_ft_model, freeze_model, unfreeze_model, make_cola, \
    make_noise_scheduler
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
    dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
    model, tokenizer = make_model(cfg['model_name'], 'unet')
    dataset = process_dataset(dataset, tokenizer)
    data_loader = make_data_loader(dataset, tokenizer, cfg['model_name'])
    result = resume(os.path.join(checkpoint_path, 'model'), resume_mode=cfg['resume_mode'])
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']}, tokenizer)
    logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if result is None:
        cfg['epoch'] = 1
        model = make_ft_model(model)
        freeze_model(model)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        cola_base = make_cola(model, cfg['cola']['model_name'])
        model.load_cola_base(cola_base)
        optimizer = {}
        scheduler = {}
        num_params = 0
        for k in cola_base:
            for n, p in cola_base[k].named_parameters():
                num_params += p.numel()
            cola_base[k] = cola_base[k].to(cfg['device_cola'])
            cola_param_k = cola_base[k].parameters()
            optimizer[k] = make_optimizer(cola_param_k, 'cola')
            scheduler[k] = make_scheduler(optimizer[k], 'cola')
        print("Number of ColA trainable parameters: {}".format(num_params))
    else:
        cfg['epoch'] = result['epoch']
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_path, 'adapter'), is_trainable=True)
        freeze_model(model)
        model = model.to(cfg['device'])
        model.print_trainable_parameters()
        cola_base = make_cola(model, cfg['cola']['model_name'])
        for k in cola_base:
            cola_base[k].load_state_dict(result['cola_base_state_dict'][k])
        model.load_cola_base(cola_base)
        optimizer = {}
        scheduler = {}
        num_params = 0
        for k in cola_base:
            for n, p in cola_base[k].named_parameters():
                num_params += p.numel()
            cola_base[k] = cola_base[k].to(cfg['device_cola'])
            cola_param_k = cola_base[k].parameters()
            optimizer[k] = make_optimizer(cola_param_k, 'cola')
            scheduler[k] = make_scheduler(optimizer[k], 'cola')
            optimizer[k].load_state_dict(result['optimizer_state_dict'][k])
            scheduler[k].load_state_dict(result['scheduler_state_dict'][k])
        print("Number of ColA trainable parameters: {}".format(num_params))
        metric.load_state_dict(result['metric_state_dict'])
        logger.load_state_dict(result['logger_state_dict'])
    vae, _ = make_model(cfg['model_name'], 'vae')
    vae = vae.to(cfg['device'])
    text_encoder, _ = make_model(cfg['model_name'], 'text_encoder')
    text_encoder = text_encoder.to(cfg['device'])
    noise_scheduler = make_noise_scheduler(model_name)
    for epoch in range(cfg['epoch'], cfg[cfg['model_name']]['num_epochs'] + 1):
        cfg['epoch'] = epoch
        train(data_loader['train'], model, vae, text_encoder, cola_base, optimizer, scheduler, noise_scheduler,
              metric, logger)
        result = {'cfg': cfg, 'epoch': cfg['epoch'] + 1,
                  'cola_base_state_dict': {k: cola_base[k].state_dict() for k in cola_base},
                  'optimizer_state_dict': {k: optimizer[k].state_dict() for k in optimizer},
                  'scheduler_state_dict': {k: scheduler[k].state_dict() for k in scheduler},
                  'metric_state_dict': metric.state_dict(), 'logger_state_dict': logger.state_dict()}
        save(result, os.path.join(checkpoint_path, 'model'))
        model.save_pretrained(os.path.join(checkpoint_path, 'adapter'))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            makedir_exist_ok(best_path)
            shutil.copy(os.path.join(checkpoint_path, 'model'), os.path.join(best_path, 'model'))
            shutil.copytree(os.path.join(checkpoint_path, 'adapter'), os.path.join(best_path, 'adapter'),
                            dirs_exist_ok=True)
        logger.reset()
    return


def train(data_loader, unet, vae, text_encoder, cola_base, optimizer, scheduler, noise_scheduler, metric, logger):
    unet.train(True)
    start_time = time.time()
    input_buffer = defaultdict(list)
    output_target_buffer = defaultdict(list)
    for i, input in enumerate(data_loader):
        for k in cola_base:
            lr = optimizer[k].param_groups[0]['lr']
            cola_base[k] = cola_base[k].to(cfg['device'])
            cola_base[k].train(False)
            freeze_model(cola_base[k])
        if cfg['test_computation']:
            s = time.time()
        input = to_device(input, cfg['device'])
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

        output_ = {'Loss': loss.detach().item()}
        input_size = input['input_ids'].size(0) / 2
        loss.backward()
        unet.zero_grad()
        input_i, output_target_i = unet.flush()
        if cfg['test_computation']:
            cfg['time_used'].append(time.time() - s)
        for k in input_i:
            input_buffer[k].append(input_i[k])
            output_target_buffer[k].append(output_target_i[k])
        if (i + 1) % cfg['cola']['num_steps'] == 0:
            for k in input_buffer:
                if cfg['test_computation']:
                    s = time.time()
                cola_base[k] = cola_base[k].to(cfg['device_cola'])
                unfreeze_model(cola_base[k])
                input_cola = torch.cat(input_buffer[k], dim=0)
                output_target_cola = torch.cat(output_target_buffer[k], dim=0)
                input_cola = {'data': input_cola, 'target': output_target_cola}
                cola_base[k].fit(input_cola, optimizer[k], scheduler[k])
                if cfg['test_computation']:
                    cfg['time_used_cola'].append(time.time() - s)
            input_buffer = defaultdict(list)
            output_target_buffer = defaultdict(list)
        evaluation = metric.evaluate('train', 'batch', None, output_)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - cfg['epoch']) * batch_time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train')
            print(logger.write('train', ['loss']), flush=True)
        if cfg['test_computation']:
            mem_free, mem_total = torch.cuda.mem_get_info(cfg['device'])
            cfg['mem_used'].append(mem_total - mem_free)
            if cfg['device_cola'] != 'cpu':
                mem_free_cola, mem_total_cola = torch.cuda.mem_get_info(cfg['device_cola'])
                cfg['mem_used_cola'].append(mem_total_cola - mem_free_cola)
            if i == cfg['num_test_iter']:
                print(cfg['time_used'])
                print(cfg['mem_used'])
                print(cfg['time_used_cola'])
                print(cfg['mem_used_cola'])
                print('Run time backward: {}({})'.format(np.mean(cfg['time_used'][1:]),
                                                         np.std(cfg['time_used'][1:])))
                print('Run time (ColA, M={}): {}({})'.format(len(list(cola_base.keys())),
                                                             np.mean(cfg['time_used_cola'][1:]),
                                                             np.std(cfg['time_used_cola'][1:])))
                print('Memory used: {}/({})'.format(np.mean(cfg['mem_used'][1:]),
                                                    np.std(cfg['mem_used'][1:])))
                if cfg['device_cola'] != 'cpu':
                    print('Memory used (ColA): {}/{}'.format(np.mean(cfg['mem_used_cola'][1:]),
                                                             np.std(cfg['mem_used_cola'][1:])))
                print('-----------------')
                exit()
    return


if __name__ == "__main__":
    main()

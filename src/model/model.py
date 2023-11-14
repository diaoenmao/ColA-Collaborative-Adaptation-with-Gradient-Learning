import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
from torchvision import transforms
from transformers import get_linear_schedule_with_warmup
from config import cfg
from diffusers import DDPMScheduler
from .huggingface import make_hf_model
from module.peft import get_peft_model, TaskType, LoraConfig, AdaLoraConfig, IA3Config, PromptTuningInit, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig, ColaConfig, UNET_TO_COLA_TARGET_MODULES_MAPPING, \
    UNET_TO_LORA_TARGET_MODULES_MAPPING


def make_model(model_name, sub_model_name=None):
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i']:
        model, tokenizer = make_hf_model(model_name, sub_model_name)
    else:
        model = eval('model.{}()'.format(model_name))
        tokenizer = None
    return model, tokenizer


def make_loss(output, input):
    if 'target' in input:
        loss = loss_fn(output['target'], input['target'])
    else:
        return
    return loss


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = kld_loss(output, target, reduction=reduction)
    return loss


def cross_entropy_loss(output, target, reduction='mean'):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction=reduction)
    return ce


def kld_loss(output, target, reduction='batchmean'):
    kld = F.kl_div(F.log_softmax(output, dim=-1), target, reduction=reduction)
    return kld


def mse_loss(output, target, reduction='mean'):
    mse = F.mse_loss(output, target, reduction=reduction)
    return mse


def init_param(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def make_optimizer(parameters, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                                weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


class NoOpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(NoOpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = NoOpScheduler(optimizer)
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_steps']['train'] *
                                                                          cfg[cfg['model_name']]['num_epochs'],
                                                         eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    elif cfg[tag]['scheduler_name'] == 'LinearAnnealingLR':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
            cfg['num_steps']['train'] * cfg[cfg['model_name']]['num_epochs'] * cfg[tag]['warmup_ratio']),
                                                    num_training_steps=cfg['num_steps']['train'] *
                                                                       cfg[cfg['model_name']]['num_epochs'])
    elif cfg[tag]['scheduler_name'] == 'ConstantLR':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=cfg[tag]['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler

def make_noise_scheduler(tag):
    if 'noise_scheduler_name' not in cfg[tag]:
        raise ValueError('Not valid noise scheduler name')

    if cfg[tag]['noise_scheduler_name'] == 'DDPM':
        noise_scheduler = DDPMScheduler(
            beta_start=cfg[tag]['beta_start'],
            beta_end=cfg[tag]['beta_end'],
            beta_schedule=cfg[tag]['beta_schedule'],
            num_train_timesteps=cfg[tag]['num_train_timesteps'],
        )
    else:
        raise ValueError('Not valid noise scheduler name')
    return noise_scheduler 

def make_ft_model(model):
    if cfg['task_name'] == 'clm':
        peft_config = make_config_clm()
    elif cfg['task_name'] == 's2s':
        peft_config = make_config_s2s()
    elif cfg['task_name'] == 'sc':
        peft_config = make_config_sc()
    elif cfg['task_name'] == 't2i':
        peft_config = make_config_t2i()
    elif cfg['task_name'] == 'ic':
        peft_config = make_config_ic(model)
    else:
        raise ValueError('Not valid task name')
    model = get_peft_model(model, peft_config)
    return model


def freeze_model(model):
    if cfg['ft_name'] == 'cola':
        for n, p in model.named_parameters():
            p.requires_grad = False
    return

def make_config_t2i():
    model_name = cfg['model_name']
    if cfg['ft_name'] == 'dreamboothlora':
        peft_config = LoraConfig(
            r=cfg[model_name]['lora_r'],
            lora_alpha=cfg[model_name]['lora_alpha'],
            target_modules=UNET_TO_LORA_TARGET_MODULES_MAPPING,
            lora_dropout=cfg[model_name]['lora_dropout'],
            bias=cfg[model_name]['lora_bias'],
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'dreamboothcola':
        peft_config = ColaConfig(
            target_modules=UNET_TO_COLA_TARGET_MODULES_MAPPING,
            inference_mode=False,
        )
    else:
        raise ValueError('Not valid ft name')
    return peft_config

def unfreeze_model(model):
    if cfg['ft_name'] == 'cola':
        for n, p in model.named_parameters():
            p.requires_grad = True
    return


def make_delta_weight(cola_base):
    with torch.no_grad():
        delta_weight = {}
        for k in cola_base:
            delta_weight[k] = cola_base[k].make_delta_weight()
            if isinstance(delta_weight[k], tuple):
                delta_weight_0, delta_weight_1 = delta_weight[k]
                delta_weight_0 = delta_weight_0.to('cpu')
                delta_weight_1 = delta_weight_1.to('cpu')
                delta_weight[k] = (delta_weight_0, delta_weight_1)
            else:
                delta_weight[k] = delta_weight[k].to('cpu')
    return delta_weight


def make_config_clm():
    if cfg['ft_name'] == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'adalora':
        peft_config = AdaLoraConfig(
            init_r=64,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            deltaT=10,
            lora_alpha=8,
            lora_dropout=0.0,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'ia3':
        peft_config = IA3Config(task_type=TaskType.CAUSAL_LM, inference_mode=False, feedforward_modules=[])
    elif cfg['ft_name'] == 'promptune':
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=20,
            prompt_tuning_init_text="Label: ",
            tokenizer_name_or_path=cfg['tokenizer_name_or_path'],
        )
    elif cfg['ft_name'] == 'prefixtune':
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
    elif cfg['ft_name'] == 'ptune':
        peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=20,
                                          encoder_hidden_size=128)
    elif cfg['ft_name'] == 'cola':
        peft_config = ColaConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False)
    else:
        raise ValueError('Not valid ft name')
    return peft_config


def make_config_s2s():
    if cfg['ft_name'] == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'adalora':
        peft_config = AdaLoraConfig(
            init_r=64,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            deltaT=10,
            lora_alpha=8,
            lora_dropout=0.0,
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'ia3':
        peft_config = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, feedforward_modules=[])
    elif cfg['ft_name'] == 'promptune':
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=20,
            prompt_tuning_init_text="Label: ",
            inference_mode=False,
            tokenizer_name_or_path=cfg['tokenizer_name_or_path'],
        )
    elif cfg['ft_name'] == 'prefixtune':
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)
    elif cfg['ft_name'] == 'ptune':
        peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20,
                                          encoder_hidden_size=128)
    elif cfg['ft_name'] == 'cola':
        peft_config = ColaConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False)
    else:
        raise ValueError('Not valid ft name')
    return peft_config


def make_config_sc():
    if cfg['ft_name'] == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'adalora':
        peft_config = AdaLoraConfig(
            init_r=64,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            deltaT=10,
            lora_alpha=8,
            lora_dropout=0.0,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'ia3':
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS, inference_mode=False, feedforward_modules=[])
    elif cfg['ft_name'] == 'promptune':
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=20,
            prompt_tuning_init_text="Label: ",
            inference_mode=False,
            tokenizer_name_or_path=cfg['tokenizer_name_or_path'],
        )
    elif cfg['ft_name'] == 'prefixtune':
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, num_virtual_tokens=20)
    elif cfg['ft_name'] == 'ptune':
        peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, num_virtual_tokens=20,
                                          encoder_hidden_size=128)
    elif cfg['ft_name'] == 'cola':
        peft_config = ColaConfig(task_type=TaskType.SEQ_CLS, inference_mode=False)
    else:
        raise ValueError('Not valid ft name')
    return peft_config


def make_config_ic(model):
    target_modules = []
    for k, v in model.named_modules():
        if isinstance(v, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            target_modules.append(k)
    if cfg['ft_name'] == 'lora':
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            inference_mode=False,
        )
    elif cfg['ft_name'] == 'cola':
        peft_config = ColaConfig(target_modules=target_modules, inference_mode=False)
    else:
        raise ValueError('Not valid ft name')
    return peft_config

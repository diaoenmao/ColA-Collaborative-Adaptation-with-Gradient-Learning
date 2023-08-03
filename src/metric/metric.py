import torch
import torch.nn.functional as F
from config import cfg
from module import recur


def make_metric(metric_name):
    if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
        pivot = -float('inf')
        pivot_direction = 'up'
        pivot_name = 'Accuracy'
    if cfg['task_name'] == 'clm':
        if cfg['data_name'] in ['raft']:
            pivot = float('inf')
            pivot_direction = 'down'
            pivot_name = 'Perplexity'
            for k in metric_name:
                metric_name[k].extend(['Perplexity'])
        else:
            raise ValueError('Not valid data name')
    elif cfg['task_name'] == 's2s':
        if cfg['data_name'] in ['fpb']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'Accuracy'
            for k in metric_name:
                metric_name[k].extend(['Accuracy'])
        else:
            raise ValueError('Not valid data name')
    else:
        raise ValueError('Not valid task name')
    metric = Metric(metric_name, pivot, pivot_direction, pivot_name)
    return metric


def Perplexity(output):
    ppl = output.exp().item()
    return ppl


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, -1, True, True)[1]).view(-1)
        batch_size = torch.numel(target)
        pred_k = output.topk(topk, -1, True, True)[1]
        correct_k = pred_k.eq(target.unsqueeze(-1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse


class Metric(object):
    def __init__(self, metric_name, pivot, pivot_direction, pivot_name):
        self.pivot, self.pivot_name, self.pivot_direction = pivot, pivot_name, pivot_direction
        self.metric_name = self.make_metric_name(metric_name)
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Perplexity': (lambda input, output: recur(Perplexity, output['loss'])),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['target'], input['target'])),
                       'RMSE': (lambda input, output: recur(RMSE, output['target'], input['target']))}

    def make_metric_name(self, metric_name):
        return metric_name

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

    def load_state_dict(self, state_dict):
        self.pivot = state_dict['pivot']
        self.pivot_name = state_dict['pivot_name']
        self.pivot_direction = state_dict['pivot_direction']
        return

    def state_dict(self):
        return {'pivot': self.pivot, 'pivot_name': self.pivot_name, 'pivot_direction': self.pivot_direction}

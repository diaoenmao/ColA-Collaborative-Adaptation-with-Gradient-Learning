import torch
import torch.nn.functional as F
import evaluate
from collections import defaultdict
from config import cfg
from module import recur


def make_metric(metric_name):
    if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
        pivot = -float('inf')
        pivot_direction = 'up'
        pivot_name = 'Accuracy'
        for k in metric_name:
            metric_name[k].extend(['Accuracy'])
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
                # metric_name[k].extend(['Rouge'])
        else:
            raise ValueError('Not valid data name')
    elif cfg['task_name'] == 'sc':
        if cfg['data_name'] in ['glue']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'GLUE'
            metric_name['test'].extend(['GLUE'])
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


class GLUE:
    def __init__(self, subset_name):
        self.metric = evaluate.load('glue', subset_name)

    def add(self, input, output):
        predictions = output['target'].argmax(dim=-1)
        references = input['target']
        self.metric.add_batch(predictions=predictions, references=references)
        return

    def __call__(self, *args, **kwargs):
        glue = self.metric.compute()
        glue = sum(glue.values()) / len(glue)
        return glue


class Metric:
    def __init__(self, metric_name, pivot, pivot_direction, pivot_name):
        self.pivot, self.pivot_name, self.pivot_direction = pivot, pivot_name, pivot_direction
        self.metric_name = metric_name
        self.metric = self.make_metric(metric_name)

    def make_metric(self, metric_name):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'Perplexity':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input,
                                                                           output: recur(Perplexity, output['loss']))}
                elif m == 'Accuracy':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(Accuracy, output['target'], input['target']))}
                elif m == 'RMSE':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(RMSE, output['target'], input['target']))}
                elif m == 'GLUE':
                    metric[split][m] = {'mode': 'full',
                                        'metric': GLUE(cfg['hf_subset_name'])}
                else:
                    raise ValueError('Not valid metric name')
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input=None, output=None, metric_name=None):
        metric_name = self.metric_name if metric_name is None else metric_name
        evaluation = {}
        for metric_name_ in metric_name[split]:
            if self.metric[split][metric_name_]['mode'] == mode:
                evaluation[metric_name_] = self.metric[split][metric_name_]['metric'](input, output)
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

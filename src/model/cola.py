import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from config import cfg
from .model import init_param, mse_loss
from module import to_device
from module.peft.tuners.cola import ColaLayer

warnings.filterwarnings('ignore', category=ConvergenceWarning)


class LowRank(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.cola_A = nn.Linear(input_size, hidden_size, bias=False)
        self.cola_B = nn.Linear(hidden_size, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.cola_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.cola_B.weight)
        return

    def fit(self, input, optimizer, scheduler):
        self.train(True)
        input = to_device(input, cfg['device'])
        output = {}
        x = input['data']
        output['target'] = self.forward(x)
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'])
        output['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        return output

    def make_delta_weight(self):
        return self.cola_B.weight.data @ self.cola_A.weight.data

    def forward(self, x):
        x = self.dropout(x)
        x = self.cola_A(x)
        x = self.cola_B(x)
        return x


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)
        return

    def fit(self, input, optimizer, scheduler):
        self.train(True)
        input = to_device(input, cfg['device'])
        output = {}
        x = input['data']
        output['target'] = self.forward(x)
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'])
        output['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        return output

    def make_delta_weight(self):
        return self.linear.weight.data

    def forward(self, x):
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, num_layers, activation, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        blocks = []
        for _ in range(num_layers):
            blocks.append(nn.Linear(input_size, hidden_size))
            if activation == 'relu':
                blocks.append(nn.ReLU())
            elif activation == 'sigmoid':
                blocks.append(nn.Sigmoid())
            else:
                raise ValueError('Not valid activation')
            input_size = hidden_size
            hidden_size = int(hidden_size * scale_factor)
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(input_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            nn.init.zeros_(p)
        return

    def fit(self, input, optimizer, scheduler):
        self.train(True)
        input = to_device(input, cfg['device'])
        output = {}
        x = input['data']
        output['target'] = self.forward(x)
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'])
        output['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        return output

    def make_delta_weight(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.blocks(x)
        x = self.linear(x)
        return x


class SK(nn.Module):
    def __init__(self, input_size, output_size, model_name):
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
        if model_name == 'skmlp':
            model = MLPRegressor(max_iter=1, warm_start=True)

        else:
            raise ValueError('Not valid model name')
        self.model = model
        self.reset_parameters()

    def reset_parameters(self):
        data = np.zeros([1, self.input_size])
        target = np.zeros([1, self.output_size])
        self.model.fit(data, target)
        for i in range(len(self.model.coefs_)):
            self.model.coefs_[i] = np.zeros(self.model.coefs_[i].shape)
            self.model.intercepts_[i] = np.zeros(self.model.intercepts_[i].shape)
        return

    def state_dict(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model = state_dict
        return

    def fit(self, input, optimizer, scheduler):
        output = {}
        data, target = input['data'].cpu().numpy(), input['target'].cpu().numpy()
        data = data.reshape(-1, self.input_size)
        target = target.reshape(-1, self.output_size)
        lr = optimizer.param_groups[0]['lr']
        if lr > 0:
            self.model.set_params(learning_rate_init=lr)
            self.model.fit(data, target)
        output_target = self.model.predict(data)
        output_target = output_target.reshape(input['target'].shape)
        output['target'] = input['target'].new_tensor(output_target)
        output['loss'] = 0.5 * F.mse_loss(output['target'], input['target'], reduction='sum')
        scheduler.step()
        return output

    def make_delta_weight(self):
        raise NotImplementedError

    def forward(self, input):
        x = input.cpu().numpy()
        x = x.reshape(-1, self.input_size)
        try:
            x = self.model.predict(x)
            x = x.reshape((*input.shape[:-1], self.output_size))
        except NotFittedError:
            x = np.zeros((*input.shape[:-1], self.output_size))
        x = input.new_tensor(x)
        return x


class Router(nn.Module):
    def __init__(self, model, dist_mode):
        super().__init__()
        self.model = nn.ModuleList(model)
        self.dist_mode = dist_mode
        self.split = None
        self.unique_split = None
        self.indices = None
        self.sorted_indices = None

    def make_split(self, split):
        self.split = split
        self.unique_split = torch.unique(split)
        indices = []
        for unique_value in self.unique_split:
            mask_i = self.split == unique_value
            indices_i = torch.nonzero(mask_i).view(-1)
            indices.append(indices_i)
        self.indices = indices
        self.sorted_indices = torch.argsort(torch.cat(indices))
        return

    def fit(self, input, optimizer=None, scheduler=None):
        if self.dist_mode == 'alone':
            for i in range(len(self.unique_split)):
                data_i, target_i = input['data'][self.indices[i]], input['target'][self.indices[i]]
                input_i = {'data': data_i, 'target': target_i}
                self.model[self.unique_split[i]].fit(input_i, optimizer[i], scheduler[i])
        elif self.dist_mode == 'col':
            for i in range(len(self.unique_split)):
                data_i, target_i = input['data'][self.indices[i]], input['target'][self.indices[i]]
                input_i = {'data': data_i, 'target': target_i}
                self.model[self.unique_split[i]].fit(input_i, optimizer[i], scheduler[i])
        else:
            raise ValueError('Not valid dist mode')
        return

    def make_delta_weight(self):
        delta_weight = []
        for i in range(len(self.model)):
            delta_weight_i = self.model[i].make_delta_weight()
            delta_weight.append(delta_weight_i)
        delta_weight = torch.stack(delta_weight, dim=0).mean(dim=0)
        return delta_weight

    def forward(self, x):
        if self.dist_mode == 'alone':
            x_ = []
            for i in range(len(self.unique_split)):
                x_i = x[self.indices[i]]
                x_i = self.model[self.unique_split[i]](x_i)
                x_.append(x_i)
            x_ = torch.cat(x_, dim=0)
            x = x_[self.sorted_indices]
        elif self.dist_mode == 'col':
            x_ = []
            for i in range(len(self.unique_split)):
                x_i = x[self.indices[i]]
                x_i_ = []
                for j in range(len(self.model)):
                    x_i_j = self.model[i](x_i)
                    if j != self.unique_split[i]:
                        x_i_j = x_i_j.detach()
                    x_i_.append(x_i_j)
                x_i = torch.stack(x_i_, dim=0).mean(dim=0)
                x_.append(x_i)
            x_ = torch.cat(x_, dim=0)
            x = x_[self.sorted_indices]
        else:
            raise ValueError('Not valid dist mode')
        return x


def make_cola_model(model_name, input_size, output_size):
    if model_name == 'lowrank':
        hidden_size = cfg['cola']['lowrank']['hidden_size']
        dropout = cfg['cola']['lowrank']['dropout']
        model = LowRank(input_size, hidden_size, output_size, dropout)
        model.apply(init_param)
    elif model_name == 'linear':
        model = Linear(input_size, output_size)
        model.apply(init_param)
    elif model_name == 'mlp':
        hidden_size = cfg['cola']['mlp']['hidden_size']
        scale_factor = cfg['cola']['mlp']['scale_factor']
        num_layers = cfg['cola']['mlp']['num_layers']
        activation = cfg['cola']['mlp']['activation']
        model = MLP(input_size, hidden_size, scale_factor, num_layers, activation, output_size)
        model.apply(init_param)
    elif model_name in ['skmlp']:
        model = SK(input_size, output_size, model_name)
    else:
        raise ValueError('Not valid model name')
    return model


def make_model_name(model_name):
    model_name_list = model_name.split("~")
    num_model_each = cfg['num_split'] // len(model_name_list)
    remainder = cfg['num_split'] % len(model_name_list)
    model_name_ = []
    for model_name_i in model_name_list:
        model_name_.extend([model_name_i] * num_model_each)
    for i in range(remainder):
        model_name_.append(model_name_list[i])
    return model_name_


def make_cola(model, model_name, dist_mode='joint'):
    if dist_mode in ['alone', 'col']:
        cfg['cola']['model_name'] = make_model_name(model_name)
    cola = {}
    for name, module in model.base_model.named_modules():
        if 'original_module' not in name and isinstance(module, ColaLayer):
            input_size = module.in_features
            output_size = module.out_features
            if dist_mode in ['alone', 'col']:
                cola[name] = []
                for i in range(cfg['num_split']):
                    cola_model_i = make_cola_model(cfg['cola']['model_name'][i], input_size, output_size)
                    cola[name].append(cola_model_i)
                cola[name] = Router(cola[name], dist_mode)
            else:
                cola[name] = make_cola_model(model_name, input_size, output_size)
    return cola

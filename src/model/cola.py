import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from config import cfg
from .model import init_param, mse_loss
from module.peft.tuners.cola import ColaLayer

warnings.filterwarnings('ignore', category=ConvergenceWarning)


class LR(nn.Module):
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

    def f(self, input):
        output = {}
        x = input['data']
        x = self.forward(x)
        output['target'] = x
        output['loss'] = F.mse_loss(output['target'], input['target'])
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

    def f(self, input):
        output = {}
        x = input['data']
        x = self.forward(x)
        output['target'] = x
        output['loss'] = F.mse_loss(output['target'], input['target'])
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

    def f(self, input):
        output = {}
        x = input['data']
        x = self.forward(x)
        output['target'] = x
        output['loss'] = F.mse_loss(output['target'], input['target'])
        return output

    def make_delta_weight(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.blocks(x)
        x = self.linear(x)
        return x


class SK(nn.Module):
    def __init__(self, input_size, output_size, model_name, max_iter=1):
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
        self.max_iter = max_iter
        if model_name == 'skmlp':
            model = MLPRegressor(max_iter=self.max_iter, warm_start=True)
        else:
            raise ValueError('Not valid model name')
        self.model = model

    def state_dict(self):
        return self.model.get_params()

    def load_state_dict(self, state_dict):
        self.model.set_params(**state_dict)
        return

    def fit(self, input):
        output = {}
        data, target = input['data'].cpu().numpy(), input['target'].cpu().numpy()
        data = data.reshape(-1, self.input_size)
        target = target.reshape(-1, self.output_size)
        self.model.fit(data, target)
        output_target = self.model.predict(data)
        output_target = output_target.reshape(input['target'].shape)
        output['target'] = input['target'].new_tensor(output_target)
        output['loss'] = F.mse_loss(output['target'], input['target'])
        return output

    def predict(self, input):
        output = {}
        data = input['data'].cpu().numpy()
        data = data.reshape(-1, data.shape[-1])
        output_target = self.model.predict(data)
        output_target = output_target.reshape(input['target'].shape)
        output['target'] = input['target'].new_tensor(output_target)
        output['loss'] = F.mse_loss(output['target'], input['target'], loss_mode=input['loss_mode'])
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


def make_cola(model, model_name):
    cola = {}
    for name, module in model.base_model.named_modules():
        if isinstance(module, ColaLayer):
            input_size = module.in_features
            output_size = module.out_features
            if model_name == 'lr':
                hidden_size = cfg['cola']['model']['hidden_size']
                dropout = cfg['cola']['model']['dropout']
                cola[name] = LR(input_size, hidden_size, output_size, dropout)
                cola[name].apply(init_param)
            elif model_name == 'linear':
                cola[name] = Linear(input_size, output_size)
                cola[name].apply(init_param)
            elif model_name == 'mlp':
                hidden_size = cfg['cola']['model']['hidden_size']
                scale_factor = cfg['cola']['model']['scale_factor']
                num_layers = cfg['cola']['model']['num_layers']
                activation = cfg['cola']['model']['activation']
                cola[name] = MLP(input_size, hidden_size, scale_factor, num_layers, activation, output_size)
                cola[name].apply(init_param)
            elif model_name in ['skmlp']:
                cola[name] = SK(input_size, output_size, model_name, max_iter=cfg['cola']['num_epochs'])
            else:
                raise ValueError('Not valid model name')
    return cola

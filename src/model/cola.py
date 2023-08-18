import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .model import init_param, mse_loss
from module.peft.tuners.cola import ColaLayer


class ColA(nn.Module):
    def __init__(self, name, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.name = name
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.cola_A = nn.Linear(input_size, hidden_size)
        self.cola_B = nn.Linear(hidden_size, output_size)

    def f(self, input):
        output = {}
        x = input['data']
        x = self.forward(x)
        output['target'] = x
        output['loss'] = F.mse_loss(output['target'], input['target'])
        return output

    def forward(self, x):
        x = self.dropout(x)
        x = self.cola_A(x)
        # x = F.relu(x)
        x = self.cola_B(x)
        return x


def make_cola(model, mode=None):
    cola = {}
    for name, module in model.base_model.named_modules():
        if isinstance(module, ColaLayer):
            input_size = module.in_features
            hidden_size = cfg['cola']['hidden_size']
            output_size = module.out_features
            dropout = cfg['cola']['dropout']
            cola[name] = ColA(name, input_size, hidden_size, output_size, dropout)
            cola[name].apply(init_param)
    return cola

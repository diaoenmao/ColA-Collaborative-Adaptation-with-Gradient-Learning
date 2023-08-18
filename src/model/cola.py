import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .model import init_param, mse_loss
from module.peft.tuners.cola import ColaLayer


class ColA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cola_A = nn.Linear(input_size, hidden_size)
        self.cola_B = nn.Linear(hidden_size, output_size)
        # self.cola_A = nn.Linear(input_size, output_size)

    def f(self, input):
        output = {}
        x = input['data']
        x = self.forward(x)
        output['target'] = x
        output['loss'] = mse_loss(output['target'], input['target'])
        # output['loss'] = -(F.normalize(output['target'], dim=-1) * F.normalize(input['target'], dim=-1)).sum(dim=-1).mean()
        return output

    def forward(self, x):
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
            cola[name] = ColA(input_size, hidden_size, output_size)
            cola[name].apply(init_param)
    return cola

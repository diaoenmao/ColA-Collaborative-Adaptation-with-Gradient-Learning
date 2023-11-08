import argparse
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import defaultdict
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from metric import make_metric, make_logger
from model import make_model, make_optimizer, make_scheduler, make_ft_model, freeze_model, make_cola
from module import save, to_device, process_control, resume, makedir_exist_ok, PeftModel


def verify_gl():
    grad_pool = []

    def backward_hook(grad):
        grad_ = grad.detach()
        grad_pool.append(grad_)
        return

    N = 1
    D_1 = 5
    D_2 = 5
    S = 1
    C = 2

    # input_1 = torch.arange(N * D_1).float().view(N, D_1)
    # weight_1 = torch.arange(D_1 * D_2).float().view(D_1, D_2)
    # weight_2 = torch.arange(D_2 * C).float().view(D_2, C)
    input_1 = torch.randn((N, D_1))
    weight_1 = torch.randn((D_1, D_2))
    weight_2 = torch.randn((D_2, C))
    weight_1.requires_grad = True
    weight_2.requires_grad = True

    output_1 = F.linear(input_1, weight_1.t())
    output_1.register_hook(backward_hook)
    input_2 = F.relu(output_1)

    # input_2 = output_1
    output_2 = F.linear(input_2, weight_2.t())
    output_2.register_hook(backward_hook)
    target = torch.zeros((N,), dtype=torch.long)

    loss = F.cross_entropy(output_2, target, reduction='mean')
    print(input_1)
    print(weight_1)
    print(output_1)
    print(input_2)
    print(weight_2)
    print(output_2)
    print(loss)
    print('-----------------')
    loss.backward()
    weight_1_grad = weight_1.grad
    weight_2_grad = weight_2.grad
    output_1_grad = grad_pool[1]
    output_2_grad = grad_pool[0]
    print(weight_1_grad)
    print(weight_2_grad)
    print(output_1_grad)
    print(output_2_grad)
    print('-----------------')
    input_1_gl = input_1.clone().detach()
    output_1_gl = output_1.clone().detach()
    weight_1_gl = weight_1.clone().detach()
    weight_1_gl.requires_grad = True
    output_1_gl_ = F.linear(input_1_gl, weight_1_gl.t())
    gl_loss_1 = 0.5 * F.mse_loss(output_1_gl_, output_1_gl - output_1_grad.detach(), reduction='sum')
    gl_loss_1.backward()
    weight_1_gl_grad = weight_1_gl.grad
    print(weight_1_gl_grad)
    print(torch.allclose(weight_1_gl_grad, weight_1_grad, atol=1e-05))
    print('-----------------')
    input_2_gl = input_2.clone().detach()
    output_2_gl = output_2.clone().detach()
    weight_2_gl = weight_2.clone().detach()
    weight_2_gl.requires_grad = True
    output_2_gl_ = F.linear(input_2_gl, weight_2_gl.t())
    gl_loss_2 = 0.5 * F.mse_loss(output_2_gl_, output_2_gl - output_2_grad.detach(), reduction='sum')
    gl_loss_2.backward()
    weight_2_gl_grad = weight_2_gl.grad
    print(weight_2_gl_grad)
    print(torch.allclose(weight_2_gl_grad, weight_2_grad, atol=1e-05))
    return


def verify_cola():
    grad_pool = []

    def backward_hook(grad):
        grad_ = grad.detach()
        grad_pool.append(grad_)
        return

    N = 1
    D_1 = 5
    D_2 = 5
    S = 1
    C = 2

    # input_1 = torch.arange(N * D_1).float().view(N, D_1)
    # weight_1 = torch.arange(D_1 * D_2).float().view(D_1, D_2)
    # weight_2 = torch.arange(D_2 * C).float().view(D_2, C)
    # weight_1_aux = torch.arange(D_1 * D_2).float().view(D_1, D_2)
    # weight_2_aux = torch.arange(D_2 * C).float().view(D_2, C)
    input_1 = torch.randn((N, D_1))
    weight_1 = torch.randn((D_1, D_2))
    weight_2 = torch.randn((D_2, C))
    weight_1_aux = torch.randn((D_1, D_2))
    weight_2_aux = torch.randn((D_2, C))
    weight_1.requires_grad = False
    weight_2.requires_grad = False
    weight_1_aux.requires_grad = True
    weight_2_aux.requires_grad = True

    output_1 = F.linear(input_1, weight_1.t())
    output_1_aux = F.linear(input_1, weight_1_aux.t())
    output_1_aux.register_hook(backward_hook)
    output_1 = output_1 + output_1_aux
    output_1.register_hook(backward_hook)

    input_2 = F.relu(output_1)
    # input_2 = output_1
    output_2 = F.linear(input_2, weight_2.t())
    output_2_aux = F.linear(input_2, weight_2_aux.t())
    output_2_aux.register_hook(backward_hook)
    output_2 = output_2 + output_2_aux
    output_2.register_hook(backward_hook)

    target = torch.zeros((N,), dtype=torch.long)
    loss = F.cross_entropy(output_2, target, reduction='mean')
    print(input_1)
    print(weight_1)
    print(weight_1_aux)
    print(output_1)
    print(input_2)
    print(weight_2)
    print(weight_2_aux)
    print(output_2)
    print(loss)
    print('-----------------')
    loss.backward()
    weight_1_grad = weight_1.grad
    weight_2_grad = weight_2.grad
    weight_1_aux_grad = weight_1_aux.grad
    weight_2_aux_grad = weight_2_aux.grad
    output_1_aux_grad = grad_pool[3]
    output_1_grad = grad_pool[2]
    output_2_aux_grad = grad_pool[1]
    output_2_grad = grad_pool[0]
    print(weight_1_grad)
    print(weight_2_grad)
    print(weight_1_aux_grad)
    print(weight_2_aux_grad)
    print(output_1_grad)
    print(output_2_grad)
    print(output_1_aux_grad)
    print(output_2_aux_grad)
    print(torch.allclose(output_1_grad, output_1_aux_grad, atol=1e-05))
    print(torch.allclose(output_2_grad, output_2_aux_grad, atol=1e-05))
    print('-----------------')
    input_1_cola = input_1.clone().detach()
    output_1_cola = output_1_aux.clone().detach()
    weight_1_cola = weight_1_aux.clone().detach()
    weight_1_cola.requires_grad = True
    output_1_cola_ = F.linear(input_1_cola, weight_1_cola.t())
    cola_loss_1 = 0.5 * F.mse_loss(output_1_cola_, output_1_cola - output_1_grad.detach(), reduction='sum')
    cola_loss_1.backward()
    weight_1_cola_grad = weight_1_cola.grad
    print(weight_1_cola_grad)
    print(torch.allclose(weight_1_cola_grad, weight_1_aux_grad, atol=1e-05))
    print('-----------------')
    input_2_cola = input_2.clone().detach()
    output_2_cola = output_2_aux.clone().detach()
    weight_2_cola = weight_2_aux.clone().detach()
    weight_2_cola.requires_grad = True
    output_2_cola_ = F.linear(input_2_cola, weight_2_cola.t())
    cola_loss_2 = 0.5 * F.mse_loss(output_2_cola_, output_2_cola - output_2_grad.detach(), reduction='sum')
    cola_loss_2.backward()
    weight_2_cola_grad = weight_2_cola.grad
    print(weight_2_cola_grad)
    print(torch.allclose(weight_2_cola_grad, weight_2_aux_grad, atol=1e-05))


    print('--------------Detach--------------------')
    grad_pool = []
    weight_1_aux.grad = None
    weight_2_aux.grad = None
    weight_1.requires_grad = False
    weight_2.requires_grad = False
    weight_1_aux.requires_grad = False
    weight_2_aux.requires_grad = False
    output_1 = F.linear(input_1, weight_1.t())
    # with torch.no_grad():
    #     output_1_aux = F.linear(input_1, weight_1_aux.t()).detach()
    output_1_aux = F.linear(input_1, weight_1_aux.t())
    output_1 = output_1 + output_1_aux
    output_1.requires_grad_(True)
    output_1.register_hook(backward_hook)

    input_2 = F.relu(output_1)
    # input_2 = output_1
    output_2 = F.linear(input_2, weight_2.t())
    # with torch.no_grad():
    #     output_2_aux = F.linear(input_2, weight_2_aux.t()).detach()
    output_2_aux = F.linear(input_2, weight_2_aux.t())
    output_2 = output_2 + output_2_aux
    output_2.requires_grad_(True)
    output_2.register_hook(backward_hook)

    target = torch.zeros((N,), dtype=torch.long)
    loss = F.cross_entropy(output_2, target, reduction='mean')
    loss.backward()
    output_1_grad = grad_pool[1]
    output_2_grad = grad_pool[0]
    input_1_cola = input_1.clone().detach()
    output_1_cola = output_1_aux.clone().detach()
    weight_1_cola = weight_1_aux.clone().detach()
    weight_1_cola.requires_grad = True
    output_1_cola_ = F.linear(input_1_cola, weight_1_cola.t())
    cola_loss_1 = 0.5 * F.mse_loss(output_1_cola_, output_1_cola - output_1_grad.detach(), reduction='sum')
    cola_loss_1.backward()
    weight_1_cola_grad_detach = weight_1_cola.grad
    print(weight_1_cola_grad_detach)
    print(torch.allclose(weight_1_cola_grad_detach, weight_1_aux_grad, atol=1e-05))
    print('-----------------')
    input_2_cola = input_2.clone().detach()
    output_2_cola = output_2_aux.clone().detach()
    weight_2_cola = weight_2_aux.clone().detach()
    weight_2_cola.requires_grad = True
    output_2_cola_ = F.linear(input_2_cola, weight_2_cola.t())
    cola_loss_2 = 0.5 * F.mse_loss(output_2_cola_, output_2_cola - output_2_grad.detach(), reduction='sum')
    cola_loss_2.backward()
    weight_2_cola_grad_detach = weight_2_cola.grad
    print(weight_2_cola_grad_detach)
    print(torch.allclose(weight_2_cola_grad_detach, weight_2_aux_grad, atol=1e-05))
    return


if __name__ == "__main__":
    verify_gl()
    verify_cola()

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

    # Initialize data, target and original parameters
    N = 10
    D_1 = 200
    D_2 = 100
    C = 2
    input_1 = torch.randn((N, D_1))
    weight_1 = torch.randn((D_1, D_2))
    weight_2 = torch.randn((D_2, C))
    target = torch.randint(0, C, (N,), dtype=torch.long)

    print('--------Classical Backprop---------')
    # Original parameters require gradient
    weight_1.requires_grad = True
    weight_2.requires_grad = True
    # First layer
    output_1 = F.linear(input_1, weight_1.t())
    input_2 = F.relu(output_1)
    # Second layer
    output_2 = F.linear(input_2, weight_2.t())
    # Compute loss
    loss = F.cross_entropy(output_2, target, reduction='mean')
    loss.backward()
    print('Loss: {}'.format(loss))
    # Save gradient for comparison
    weight_1_grad = weight_1.grad
    weight_2_grad = weight_2.grad

    print('--------Gradient Learning---------')
    # Proposed method
    # Backward original loss
    # Use the same data and parameters
    input_1_gl = input_1.clone().detach()
    weight_1_gl = weight_1.clone().detach()
    weight_2_gl = weight_2.clone().detach()
    target_gl = target.clone().detach()
    # Freeze original parameters
    weight_1_gl.requires_grad = False
    weight_2_gl.requires_grad = False
    # First layer
    output_1_gl = F.linear(input_1_gl, weight_1_gl.t())
    output_1_gl.requires_grad_(True)
    # register hook to retrieve gradient of hidden representations
    output_1_gl.register_hook(backward_hook)
    input_2_gl = F.relu(output_1_gl)
    # Second layer
    output_2_gl = F.linear(input_2_gl, weight_2_gl.t())
    # register hook to retrieve gradient of hidden representations
    output_2_gl.requires_grad_(True)
    output_2_gl.register_hook(backward_hook)
    # Compute loss
    loss = F.cross_entropy(output_2_gl, target_gl, reduction='mean')
    loss.backward()
    print('Loss: {}'.format(loss))

    # Backward on original parameters
    # Use the same data and parameters
    input_1_gl = input_1.clone().detach()
    weight_1_gl = weight_1.clone().detach()
    weight_2_gl = weight_2.clone().detach()
    # Retrieve hidden representations and gradient of hidden representations
    output_1_gl = output_1_gl.clone().detach()
    input_2_gl = input_2_gl.clone().detach()
    output_2_gl = output_2_gl.clone().detach()
    output_1_grad = grad_pool[1]
    output_2_grad = grad_pool[0]
    # First layer
    weight_1_gl.requires_grad = True
    output_1_gl_ = F.linear(input_1_gl, weight_1_gl.t())
    gl_loss_1 = 0.5 * F.mse_loss(output_1_gl_, (output_1_gl - output_1_grad).detach(), reduction='sum')  # Proposition 1
    gl_loss_1.backward()
    # Second layer
    weight_2_gl.requires_grad = True
    output_2_gl_ = F.linear(input_2_gl, weight_2_gl.t())
    gl_loss_2 = 0.5 * F.mse_loss(output_2_gl_, (output_2_gl - output_2_grad).detach(), reduction='sum')  # Proposition 1
    gl_loss_2.backward()
    # Verification
    weight_1_gl_grad = weight_1_gl.grad
    weight_2_gl_grad = weight_2_gl.grad
    if_match_1 = torch.allclose(weight_1_gl_grad, weight_1_grad, atol=1e-03)
    if_match_2 = torch.allclose(weight_2_gl_grad, weight_2_grad, atol=1e-03)
    print('[Verify Proposition 1] Gradient of original parameters (first layer): {}, '
          'Gradient of original parameters (second layer): {}'.format(if_match_1, if_match_2))
    return


def verify_cola():
    grad_pool = []

    def backward_hook(grad):
        grad_ = grad.detach()
        grad_pool.append(grad_)
        return

    # Initialize data, target, original parameters and auxiliary parameters
    N = 10
    D_1 = 200
    D_2 = 100
    C = 2
    # Initialize data and weight
    input_1 = torch.randn((N, D_1))
    weight_1 = torch.randn((D_1, D_2))
    weight_2 = torch.randn((D_2, C))
    weight_1_aux = torch.randn((D_1, D_2))
    weight_2_aux = torch.randn((D_2, C))

    print('--------Classical Backprop (LoRA-like)---------')
    # Freeze original parameters and auxiliary parameters require gradient
    weight_1.requires_grad = False
    weight_2.requires_grad = False
    weight_1_aux.requires_grad = True
    weight_2_aux.requires_grad = True
    # First layer
    output_1 = F.linear(input_1, weight_1.t())
    output_1_aux = F.linear(input_1, weight_1_aux.t())
    output_1_aux.register_hook(backward_hook)
    output_1 = output_1 + output_1_aux
    output_1.register_hook(backward_hook)
    input_2 = F.relu(output_1)
    # Second Layer
    output_2 = F.linear(input_2, weight_2.t())
    output_2_aux = F.linear(input_2, weight_2_aux.t())
    output_2_aux.register_hook(backward_hook)
    output_2 = output_2 + output_2_aux
    output_2.register_hook(backward_hook)
    target = torch.zeros((N,), dtype=torch.long)
    # Compute loss
    loss = F.cross_entropy(output_2, target, reduction='mean')
    print('Loss: {}'.format(loss))
    loss.backward()
    # Save gradient for comparison
    weight_1_aux_grad = weight_1_aux.grad
    weight_2_aux_grad = weight_2_aux.grad
    # Verification
    output_1_aux_grad = grad_pool[3]
    output_1_grad = grad_pool[2]
    output_2_aux_grad = grad_pool[1]
    output_2_grad = grad_pool[0]
    if_match_h_1 = torch.allclose(output_1_grad, output_1_aux_grad, atol=1e-03)
    if_match_h_2 = torch.allclose(output_2_grad, output_2_aux_grad, atol=1e-03)
    print('[Verify Eq. 5] Gradient of original hidden and auxiliary representation (first layer) : {}, '
          'Gradient of original hidden and auxiliary representation (second layer): {}'.format(if_match_h_1,
                                                                                               if_match_h_2))  # Eq. 5

    print('--------------Collaborative Adaptation (first version with detach)--------------------')
    grad_pool = []
    # Proposed method
    # Backward original loss
    # Use the same data and parameters
    input_1_cola = input_1.clone().detach()
    weight_1_cola = weight_1.clone().detach()
    weight_2_cola = weight_2.clone().detach()
    weight_1_aux_cola = weight_1_aux.clone().detach()
    weight_2_aux_cola = weight_2_aux.clone().detach()
    target_cola = target.clone().detach()
    # Freeze original and auxiliary parameters
    weight_1_cola.requires_grad = False
    weight_2_cola.requires_grad = False
    weight_1_aux_cola.requires_grad = False
    weight_2_aux_cola.requires_grad = False
    # First layer
    output_1_cola = F.linear(input_1_cola, weight_1_cola.t())
    with torch.no_grad():
        output_1_aux_cola = F.linear(input_1, weight_1_aux_cola.t()).detach()
    output_1_cola = output_1_cola + output_1_aux_cola
    # register hook to retrieve gradient of hidden representations
    output_1_cola.requires_grad_(True)
    output_1_cola.register_hook(backward_hook)
    input_2_cola = F.relu(output_1_cola)
    # Second layer
    output_2_cola = F.linear(input_2_cola, weight_2_cola.t())
    with torch.no_grad():
        output_2_aux_cola = F.linear(input_2_cola, weight_2_aux_cola.t()).detach()
    output_2_cola = output_2_cola + output_2_aux_cola
    # register hook to retrieve gradient of hidden representations
    output_2_cola.requires_grad_(True)
    output_2_cola.register_hook(backward_hook)
    # Compute loss
    loss = F.cross_entropy(output_2_cola, target_cola, reduction='mean')
    loss.backward()
    print('Loss: {}'.format(loss))

    # Backward on original parameters
    # Use the same data and parameters
    input_1_cola = input_1.clone().detach()
    weight_1_aux_cola = weight_1_aux_cola.clone().detach()
    weight_2_aux_cola = weight_2_aux_cola.clone().detach()
    # Retrieve hidden representations and gradient of hidden representations
    output_1_aux_cola = output_1_aux_cola.clone().detach()
    input_2_cola = input_2_cola.clone().detach()
    output_2_aux_cola = output_2_aux_cola.clone().detach()
    output_1_grad = grad_pool[1]
    output_2_grad = grad_pool[0]
    # First layer
    weight_1_aux_cola.requires_grad = True
    output_1_aux_cola_ = F.linear(input_1_cola, weight_1_aux_cola.t())
    cola_loss_1 = 0.5 * F.mse_loss(output_1_aux_cola_, (output_1_aux_cola - output_1_grad).detach(),
                                   reduction='sum')  # Proposition 1
    cola_loss_1.backward()
    # Second layer
    weight_2_aux_cola.requires_grad = True
    output_2_aux_cola_ = F.linear(input_2_cola, weight_2_aux_cola.t())
    cola_loss_2 = 0.5 * F.mse_loss(output_2_aux_cola_, (output_2_aux_cola - output_2_grad).detach(),
                                   reduction='sum')  # Proposition 1
    cola_loss_2.backward()
    # Verification
    weight_1_aux_cola_grad = weight_1_aux_cola.grad
    weight_2_aux_cola_grad = weight_2_aux_cola.grad
    if_match_g_1 = torch.allclose(weight_1_aux_cola_grad, weight_1_aux_grad, atol=1e-03)
    if_match_g_2 = torch.allclose(weight_2_aux_cola_grad, weight_2_aux_grad, atol=1e-03)
    print('[Verify Proposition 1] Gradient of auxiliary parameters (first layer): {}, '
          'Gradient of auxiliary parameters (second layer): {}'.format(if_match_g_1, if_match_g_2))

    print('--------Collaborative Adaptation---------')
    grad_pool = []
    # Proposed method
    # Backward original loss
    # Use the same data and parameters
    input_1_cola = input_1.clone().detach()
    weight_1_cola = weight_1.clone().detach()
    weight_2_cola = weight_2.clone().detach()
    weight_1_aux_cola = weight_1_aux.clone().detach()
    weight_2_aux_cola = weight_2_aux.clone().detach()
    target_cola = target.clone().detach()
    # Freeze original and auxiliary parameters
    weight_1_cola.requires_grad = False
    weight_2_cola.requires_grad = False
    weight_1_aux_cola.requires_grad = False
    weight_2_aux_cola.requires_grad = False
    # First layer
    output_1_cola = F.linear(input_1_cola, weight_1_cola.t())
    output_1_aux_cola = F.linear(input_1_cola, weight_1_aux_cola.t())
    output_1_cola = output_1_cola + output_1_aux_cola
    # register hook to retrieve gradient of hidden representations
    output_1_cola.requires_grad_(True)
    output_1_cola.register_hook(backward_hook)
    input_2_cola = F.relu(output_1_cola)
    # Second layer
    output_2_cola = F.linear(input_2_cola, weight_2_cola.t())
    output_2_aux_cola = F.linear(input_2_cola, weight_2_aux_cola.t())
    output_2_cola = output_2_cola + output_2_aux_cola
    # register hook to retrieve gradient of hidden representations
    output_2_cola.requires_grad_(True)
    output_2_cola.register_hook(backward_hook)
    # Compute loss
    loss = F.cross_entropy(output_2_cola, target_cola, reduction='mean')
    loss.backward()
    print('Loss: {}'.format(loss))

    # Backward on original parameters
    # Use the same data and parameters
    input_1_cola = input_1.clone().detach()
    weight_1_aux_cola = weight_1_aux_cola.clone().detach()
    weight_2_aux_cola = weight_2_aux_cola.clone().detach()
    # Retrieve hidden representations and gradient of hidden representations
    output_1_aux_cola = output_1_aux_cola.clone().detach()
    input_2_cola = input_2_cola.clone().detach()
    output_2_aux_cola = output_2_aux_cola.clone().detach()
    output_1_grad = grad_pool[1]
    output_2_grad = grad_pool[0]
    # First layer
    weight_1_aux_cola.requires_grad = True
    output_1_aux_cola_ = F.linear(input_1_cola, weight_1_aux_cola.t())
    cola_loss_1 = 0.5 * F.mse_loss(output_1_aux_cola_, (output_1_aux_cola - output_1_grad).detach(),
                                   reduction='sum')  # Proposition 1
    cola_loss_1.backward()
    # Second layer
    weight_2_aux_cola.requires_grad = True
    output_2_aux_cola_ = F.linear(input_2_cola, weight_2_aux_cola.t())
    cola_loss_2 = 0.5 * F.mse_loss(output_2_aux_cola_, (output_2_aux_cola - output_2_grad).detach(),
                                   reduction='sum')  # Proposition 1
    cola_loss_2.backward()
    # Verification
    weight_1_aux_cola_grad = weight_1_aux_cola.grad
    weight_2_aux_cola_grad = weight_2_aux_cola.grad
    if_match_g_1 = torch.allclose(weight_1_aux_cola_grad, weight_1_aux_grad, atol=1e-03)
    if_match_g_2 = torch.allclose(weight_2_aux_cola_grad, weight_2_aux_grad, atol=1e-03)
    print('[Verify Proposition 1] Gradient of auxiliary parameters (first layer): {}, '
          'Gradient of auxiliary parameters (second layer): {}'.format(if_match_g_1, if_match_g_2))
    return


if __name__ == "__main__":
    verify_gl()
    verify_cola()

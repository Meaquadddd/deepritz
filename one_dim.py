from statistics import mean
from turtle import forward
from unittest import result
import torch
import numpy as np
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as func
"""
this file is intended for the one dim problem in Sec 4.1
-(kappa(x,Z) u'(x,Z))' = 0
area in [-1,1]
Boundary condition u(-1,Z) = 0,u(1,Z) = 1
"""

import numpy
import torch


def grad(outputs, inputs):
    '''
    求梯度的快速函数
    '''
    return ag.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), 
                   create_graph=True)

# 梯度以及原函数计算要用同一个Z
def random_field(x,n,beta,Z):
    '''
    shape of x: (num_of_sample,dim_of_features)
    shape of Z: (num_of_sample,2,n)
    '''
    coeff = torch.arange(start=1,end=n+1).cuda()
    new_x = torch.pi*coeff*x.repeat(1,n)
    part_1 = Z[:,0,:]*torch.sin(new_x)
    part_2 = Z[:,1,:]*torch.cos(new_x)
    V = torch.sum(part_1+part_2,dim=1)/torch.sqrt(torch.tensor(n))
    result = torch.exp(beta*V)
    return result.unsqueeze(-1)


def Grad_of_random_field(x,n,beta,Z):
    coeff = torch.arange(start=1,end=n+1)
    modified_Z = torch.pi*coeff*Z
    new_x = torch.pi*coeff*x.repeat(1,n)
    sin_new_x = torch.sin(new_x)
    cos_new_x = torch.cos(new_x)
    part_1 = Z[:,0,:]*sin_new_x
    part_2 = Z[:,1,:]*cos_new_x
    modified_part_1 = modified_Z[0,:]*(-sin_new_x)
    modified_part_2 = modified_Z[1,:]*cos_new_x
    V = torch.sum(part_1+part_2,dim=1)/torch.sqrt(n)
    grad_V = torch.sum(modified_part_1+modified_part_2,dim=1)/torch.sqrt(torch.tensor(n))
    result = grad_V*torch.exp(beta*V)
    return result.unsqueeze(-1)

class classic_primal_resudial(nn.Module):
    def __init__(self):
        super(classic_primal_resudial,self).__init__()
        return
    def forward(self,model,x,K,Z,n,M):
        '''
        x : spatial interior input 
        K : random field value for given x and Z
        S : spatial boundary input
        Z : stochastic term
        n : turncated number for the V
        M : number of every batch
        penalty : the penalty coefficient for the boundary condition forcing term
        boundary value
        '''
        final_input_inter = torch.cat((x,Z.reshape(M,2*n)),dim=1)
        final_input_inter.requires_grad = True
        u = model(final_input_inter)
        u_x = grad(u,final_input_inter)[0][:,0].unsqueeze(-1)
        loss_inter = torch.mean(0.5*K*u_x**2)
        return loss_inter











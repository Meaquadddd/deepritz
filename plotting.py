from cProfile import label
from unittest import result
import matplotlib.pyplot as plt
import torch
import numpy as np
'''
只有画图才需要积分。。
'''
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


def GetGaussParams_1D(num):
    if num == 1:
        X = torch.tensor([0])
        A = torch.tensor([2])
    elif num == 2:
        X = torch.tensor([np.math.sqrt(1/3), -np.math.sqrt(1/3)])
        A = torch.tensor([1, 1])
    elif num == 3:
        X = torch.tensor([np.math.sqrt(3/5), -np.math.sqrt(3/5), 0])
        A = torch.tensor([5/9, 5/9, 8/9])
    elif num == 6:
        X = torch.tensor([0.238619186081526, -0.238619186081526, 0.661209386472165, -0.661209386472165, 0.932469514199394, -0.932469514199394])
        A = torch.tensor([0.467913934574257, 0.467913934574257, 0.360761573028128, 0.360761573028128, 0.171324492415988, 0.171324492415988])
    elif num == 10:
        X = torch.tensor([0.973906528517240, -0.973906528517240, 0.433395394129334, -0.433395394129334, 0.865063366688893, -0.865063366688893, \
             0.148874338981367, -0.148874338981367, 0.679409568299053, -0.679409568299053])
        A = torch.tensor([0.066671344307672, 0.066671344307672, 0.269266719309847, 0.269266719309847, 0.149451349151147, 0.149451349151147, \
            0.295524224714896, 0.295524224714896, 0.219086362515885, 0.219086362515885])
    else:
         raise Exception(">>> Unsupported num = {} <<<".format(num))
    return X.cuda(), A.cuda()

def GaussIntegral_1D(func, a, b, num,Z):
    '''
    I do not use this..
    '''
    term1 = (b - a) / 2
    term2 = (a + b) / 2
    X,A = GetGaussParams_1D(num)
    term3 = func(term1 * X + term2,Z)
    term4 = torch.sum(A * term3)
    val = term1 * term4
    return val

def Generate_Gauss_1D_point(a,b,num):
    term1 = (b - a) / 2
    term2 = (a + b) / 2
    X,A = GetGaussParams_1D(num)
    modi_X = term1 * X + term2
    return modi_X,A


def Gauss_integral(Z,num,a,b,n,beta):
    num_sample = Z.shape[0]
    modi_X,A = Generate_Gauss_1D_point(a,b,num)
    A = A.unsqueeze(-1).cuda()
    result_list = torch.zeros(len(modi_X),num_sample).cuda()
    i=0
    for x in modi_X:
        result_list[i,:] = 1/random_field(x.repeat(num_sample,1),n,beta,Z)[:,0]
        i = i+1
    result = torch.sum(A*result_list,dim=0)*(b-a)/2
    return result

def exact_solution(Z,num,x,n,beta):
    a = -1
    b = 1
    result = Gauss_integral(Z,num,a,x,n,beta)/Gauss_integral(Z,num,a,b,n,beta)
    return result.unsqueeze(-1)


def Slice_plot(model,x,num_sample,n,num_of_gausspoint,beta):
    '''
    unfinished ... 
    n for turnacted num 
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    Z = torch.randn(num_sample,2,n).cuda()
    X = x.repeat(num_sample,1).cuda()
    input_value = torch.cat((X,Z.reshape(num_sample,2*n)),dim=1)
    u_pred = model(input_value)[:,0].detach().cpu().numpy()
    plt.title('slice plot when x = {}'.format(x))
    sns.kdeplot(u_pred,label = 'NN')
    u_real = exact_solution(Z,num_of_gausspoint,x,n,beta)[:,0].detach().cpu().numpy()
    sns.kdeplot(u_real,label = 'Exact')
    plt.legend()
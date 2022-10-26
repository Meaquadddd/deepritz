'''
this file is intended for sampling from 
1.the given area
2.the boundary from the given area 
3.the random variable Z
'''
import torch
def Interior_1d_sampler(M,left_side,right_side):
    sampler = torch.distributions.uniform.Uniform(low=left_side,high=right_side)
    x = sampler.sample(sample_shape=(M,1))
    return x

def Boundary_1d_sampler(M,left_side,right_side):
    '''
    return spatial input and corresponding value
    '''
    weight = torch.tensor([0.5,0.5])
    S = torch.multinomial(weight,M,replacement = True)
    boundary_value = S
    S = torch.where(S==0,left_side,right_side)
    return S.unsqueeze(-1).float(),boundary_value.unsqueeze(-1).float()

def latent_generator(M,n):
    Z = torch.randn(M,2,n)
    return Z

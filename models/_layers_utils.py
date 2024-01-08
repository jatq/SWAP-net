
import torch
import torch.nn as nn

class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, y):
        return x.add(y)
    
    def __repr__(self): 
        return f'{self.__class__.__name__}'
    
class Squeeze(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, x): 
        return x.squeeze(self.dim)
    
class Concat(nn.Module):
    def __init__(self, dim=1): 
        super().__init__()
        self.dim = dim
        
    def forward(self, *x): 
        return torch.cat(*x, dim=self.dim)
    
    def __repr__(self): 
        return f'{self.__class__.__name__}(dim={self.dim})'


class LN(nn.Module):
    def __init__(self, dim):
        super(LN, self).__init__()
        self.gamma = nn.Parameter(torch.ones([1,dim,1]))
        self.beta = nn.Parameter(torch.zeros([1,dim,1]))
    
    def forward(self, x):
        mean = x.mean(dim=[1,2],keepdim=True)
        var = x.var(dim=[1,2],keepdim=True)
        x = (x - mean) / (torch.sqrt(var + 1e-8))
        return x * self.gamma + self.beta

class ConvBlock(nn.Sequential):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, stride=1, padding='same', bias=None, act=nn.ReLU, act_kwargs={},  xtra=None, **kwargs):
        kernel_size = kernel_size if kernel_size else 3
        layers = []
        conv = nn.Conv1d(ni, nf,kernel_size=kernel_size, bias=bias, stride=stride, padding=padding, **kwargs)
        bn = nn.BatchNorm1d(nf)
        layers.append(conv)
        layers.append(bn)
        if act:
            layers.append(act(**act_kwargs))

        if xtra: 
            layers.append(xtra)
        super().__init__(*layers) 
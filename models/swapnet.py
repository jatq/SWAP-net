#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models._layers_utils import Add, Concat, Squeeze


class Shortcut(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv = nn.Conv1d(ni, nf, 1, bias=None)
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, x):
        return self.bn(self.conv(x))


class CBR(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv = nn.Conv1d(ni, nf, 1, bias=None)
        self.bn = nn.BatchNorm1d(nf)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class LKC(nn.Module):
    def __init__(self, dim, k1, k2, bias):
        super().__init__() 
        self.depthwise_conv = nn.Conv1d(dim, dim, k1, stride = 1, padding="same", groups=dim, bias=bias)
        self.depthwise_dila_conv = nn.Conv1d(dim, dim, k2, stride=1, padding = "same", groups=dim, dilation=(k1+1)//2, bias=bias)
        self.pointwise_conv = nn.Conv1d(dim, dim, 1, bias=bias)
        
    def forward(self, x):
        return self.pointwise_conv(
                    self.depthwise_dila_conv(
                        self.depthwise_conv(x)))


class LKI(nn.Module):
    # ni -> nf * 4
    def __init__(self, ni, nf, ks1, ks2, bias, pool_ks = 3):
        super().__init__()
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=bias) 
        self.convs = nn.ModuleList([LKC(dim = nf, k1 = k, k2 = ks2, bias=bias) for k in ks1])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(pool_ks, stride=1, padding=pool_ks//2), nn.Conv1d(ni, nf, 1, bias=bias)])
        self.concat = Concat()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return x

    
class LKI_s(nn.Module):
    def __init__(self, ni, nf, ks1, ks2, pool_ks=3):
        super().__init__()
        hidden_channels = [nf * 2**i for i in range(4)]
        out_channels = [h * 4 for h in hidden_channels]
        in_channels = [ni] + out_channels[:-1]


        self.LKI_list, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            if (i+1) % 2 == 0:   # when i is 1 or 3
                self.shortcuts.append(Shortcut(in_channels[i-1],out_channels[i]))
            self.LKI_list.append(LKI(in_channels[i], hidden_channels[i], ks1, ks2, bias=False, pool_ks=pool_ks))    
        self.add = Add()
        self.act = nn.ReLU()
    def forward(self, x):
        res = x
        for i in range(4):
            x = self.LKI_list[i](x)
            if (i + 1) % 2 == 0: 
                res = x = self.act(self.add(x, self.shortcuts[i//2](res)))
        return x
    
class SWAP(nn.Module):
    def __init__(self, c_in=3, c_out=36, nf=32, adaptive_size=25, ks1=[17, 13, 9], ks2=7, pool_ks=3):
        super().__init__()
        self.feature_extraction = LKI_s(c_in, nf, ks1, ks2, pool_ks=pool_ks)
        self.head_nf = nf * 32
        self.feture_aggregation = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size), 
                                  CBR(self.head_nf, self.head_nf//2), 
                                  CBR(self.head_nf//2, self.head_nf//4), 
                                  CBR(self.head_nf//4, c_out), 
                                  nn.AdaptiveAvgPool1d(1),
                                  Squeeze(-1))

    def forward(self, x, is_softmax=False):
        x = self.feature_extraction(x)
        logits = self.feture_aggregation(x)
        if is_softmax:
            return nn.Softmax(dim=1)(logits)
        return logits
    
    def train_loss(self, x, y):
        pred_y = self.forward(x)
        return  F.kl_div(F.log_softmax(pred_y, dim=1), y, reduction='batchmean')  # log_softmax(logits), probability
    
    def inference(self, x):
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            pred_y = self.forward(x.cuda(), is_softmax=True).cpu().numpy()
            pred_v = np.array([self.label2azimuth(y) for y in pred_y])
        return pred_v
            
    def label2azimuth(self, y):
        # transform a probability distribution to azimuth value.
        delta = 360//y.shape[-1]
        i = np.arange(0, 360, delta)
        index = np.arange(0,int(360/delta)).astype(np.int32)
        max_i = i[np.argmax(y)]
        if abs(max_i - 180) > 100:
            i = np.arange(-180, 180, delta)
            index = np.arange(int(-180/delta), int(180/delta)).astype(np.int32)
        return np.sum(y[index] * i) % 360 %360
        
if __name__ == '__main__':
    m = SWAP(3, 360, nf=32,  ks1 = [17, 11, 5], ks2=7).cuda()
    x = torch.randn(10, 3, 200)
    print(m.inference(x))

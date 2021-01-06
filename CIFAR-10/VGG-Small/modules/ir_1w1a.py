import torch.nn as nn
import torch.nn.functional as F
from . import binaryfunction
import torch
import math


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels,
                                       kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(
            bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)

        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

    def bi_weight(self):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(
            bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        return bw


class Nomal_conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, quan=False):
        super(Nomal_conv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.quan = quan

    def forward(self, input):
        w = self.weight
        a = input

        # 将权重从min,max映射到-1，1
        #nw = 2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        nw=w - ((w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1))+
            (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))/2
        nw = 2 * nw
        nw = nw /( (w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1))-
                (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))
        if self.quan == True:
            nw = binaryfunction.Quantize8bit.apply(nw)

        output = F.conv2d(a, nw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

    def true_weight(self):
        w = self.weight
        #nw = 2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        nw=w - ((w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1))+
            (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))/2
        nw = 2 * nw
        nw = nw /( (w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1))-
                (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))
        #print(nw)
        if self.quan == True:
            nw = binaryfunction.Quantize8bit.apply(nw)
        return nw


class Nomal_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,quan=True):
        super(Nomal_linear, self).__init__(in_channels,out_channels,bias)
        self.quan=quan

    def forward(self,input):
        w = self.weight
        w_bias = self.bias

        nw=2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        if(w_bias.max()-w_bias.min() ==0):
            nw_bias=w_bias
        else:
            nw_bias=2*(w_bias - (w_bias.max()+w_bias.min())/2)/(w_bias.max()-w_bias.min())

        if self.quan == True:
            nw=binaryfunction.Quantize8bit.apply(nw)
            nw_bias=binaryfunction.Quantize8bit.apply(nw_bias)
        
        output=F.linear(input,nw,nw_bias)
        return output

    def true_weight(self):
        w = self.weight
        w_bias = self.bias

        nw=2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        if(w_bias.max()-w_bias.min() ==0):
            nw_bias=w_bias
        else:
            nw_bias=2*(w_bias - (w_bias.max()+w_bias.min())/2)/(w_bias.max()-w_bias.min())

        if self.quan == True:
            nw=binaryfunction.Quantize8bit.apply(nw)
            nw_bias=binaryfunction.Quantize8bit.apply(nw_bias)
        return nw,nw_bias

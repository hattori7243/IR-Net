import torch.nn as nn
import torch.nn.functional as F
from . import binaryfunction
from modules.quantizefunction import QuantizeNbit
import torch
import math


class Normal_conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 quan=True, quan_bit=8):
        super(Normal_conv2d, self).__init__(in_channels, out_channels,
                                            kernel_size, stride, padding, dilation, groups, bias)
        self.quan = quan
        self.quan_bit = quan_bit

    def forward(self, input):
        w = self.weight
        a = input

        # 将权重从min,max映射到-1，1
        #nw = 2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        nw = w - ((w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1)) +
                  (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))/2
        nw = 2 * nw
        nw = nw / ((w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1)) -
                   (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))
        if self.quan == True:
            nw = QuantizeNbit.apply(nw, self.quan_bit)

        output = F.conv2d(a, nw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

    def true_weight(self):
        w = self.weight
        #nw = 2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        nw = w - ((w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1)) +
                  (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))/2
        nw = 2 * nw
        nw = nw / ((w.view(w.size(0), -1).max(-1)[0].view(w.size(0), 1, 1, 1)) -
                   (w.view(w.size(0), -1).min(-1)[0].view(w.size(0), 1, 1, 1)))
        # print(nw)
        if self.quan == True:
            nw = QuantizeNbit.apply(nw, self.quan_bit)
        return nw


class Normal_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, quan=True, quan_bit=8):
        super(Normal_linear, self).__init__(in_channels, out_channels, bias)
        self.quan = quan
        self.quan_bit = quan_bit

    def forward(self, input):
        w = self.weight
        w_bias = self.bias

        nw = 2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        if(w_bias.max()-w_bias.min() == 0):
            nw_bias = w_bias
        else:
            nw_bias = 2*(w_bias - (w_bias.max()+w_bias.min())/2) / \
                (w_bias.max()-w_bias.min())

        if self.quan == True:
            nw = QuantizeNbit.apply(nw, self.quan_bit)
            nw_bias = QuantizeNbit.apply(nw_bias, self.quan_bit)

        output = F.linear(input, nw, nw_bias)
        return output

    def true_weight(self):
        w = self.weight
        w_bias = self.bias

        nw = 2*(w - (w.max()+w.min())/2)/(w.max()-w.min())
        if(w_bias.max()-w_bias.min() == 0):
            nw_bias = w_bias
        else:
            nw_bias = 2*(w_bias - (w_bias.max()+w_bias.min())/2) / \
                (w_bias.max()-w_bias.min())

        if self.quan == True:
            nw = QuantizeNbit.apply(nw, self.quan_bit)
            nw_bias = QuantizeNbit.apply(nw_bias, self.quan_bit)
        return nw, nw_bias


class Quan_conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, quan=True, quan_bit=8):
        super(Quan_conv2d, self).__init__(in_channels, out_channels,
                                          kernel_size, stride, padding, dilation, groups, bias)
        self.quan = quan
        self.quan_bit = quan_bit
        self.has_sq = False
        self.offset = 0
        self.scale = 0

    def forward(self, input):
        w = self.weight
        a = input

        if self.has_sq == True and self.quan == True:
            nw = QuantizeNbit.apply(w, self.quan_bit, self.scale, self.offset)
        else:
            nw = w
        output = F.conv2d(a, nw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

    def cal_sq(self):
        w = self.weight.detach()
        self.scale = (w.max()-w.min())/(math.pow(2, self.quan_bit)-1)
        if w.min() > 0:
            self.offset = 0
        else:
            self.offset = round(((-w.min())/self.scale).item())
        self.has_sq = True

    def true_weight(self):
        w = self.weight
        if self.quan == False:
            print('quan=False')
            return w

        if self.has_sq == True and self.quan == True:
            nw = QuantizeNbit.apply(w, self.quan_bit, self.scale, self.offset)
            return nw
        else:
            print('have not cal the scale and offset, weight is origin weight.')
            return w

    def quan_weight(self):
        if self.has_sq == True:
            return self.true_weight()/self.scale+self.offset
        else:
            print('have not cal the scale and offset, quan weight is None.')
            return None


class Quan_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, quan=True, quan_bit=8):
        super(Quan_linear, self).__init__(in_channels, out_channels, bias)
        self.quan = quan
        self.quan_bit = quan_bit
        self.has_sq = False
        self.w_offset = 0
        self.w_scale = 0
        self.a_offset = 0
        self.a_scale = 0

    def forward(self, input):
        w = self.weight
        a = self.bias

        if self.has_sq == True and self.quan == True:
            nw = QuantizeNbit.apply(
                w, self.quan_bit, self.w_scale, self.w_offset)
            na = QuantizeNbit.apply(
                a, self.quan_bit, self.a_scale, self.a_offset)
        else:
            nw = w
            na = a

        output = F.linear(input, nw, na)
        return output

    def cal_sq(self):
        w = self.weight.detach()
        a = self.bias.detach()
        self.w_scale = (w.max()-w.min())/(math.pow(2, self.quan_bit)-1)
        self.a_scale = (a.max()-a.min())/(math.pow(2, self.quan_bit)-1)
        if w.min() > 0:
            self.w_offset = 0
        else:
            self.w_offset = round(((-w.min())/self.w_scale).item())
        if a.min() > 0:
            self.a_offset = 0
        else:
            self.w_offset = round(((-w.min())/self.a_scale).item())
        self.has_sq = True

    def true_weight(self):
        w = self.weight
        a = self.bias

        if self.quan == False:
            print('quan=False')
            return w, w_bias

        if self.has_sq == True and self.quan == True:
            nw = QuantizeNbit.apply(
                w, self.quan_bit, self.w_scale, self.w_offset)
            na = QuantizeNbit.apply(
                a, self.quan_bit, self.a_scale, self.a_offset)
            return nw, na
        else:
            print('have not cal the scale and offset, weight is origin weight.')
            return w, a

    def quan_weight(self):
        if self.has_sq == True:
            return self.true_weight()[0]/self.w_scale+self.w_offset,\
                self.true_weight()[1]/self.a_scale+self.a_offset
        else:
            print('have not cal the scale and offset, quan weight is None.')
            return None

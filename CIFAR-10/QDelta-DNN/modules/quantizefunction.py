from torch.autograd import Function
import torch
import torch.nn as nn
import math

class QuantizeNbit(Function):
    @staticmethod
    def forward(ctx, input,quan_bit=8,scale=0,offset=0):
        ctx.save_for_backward(input)
        num = math.pow(2,quan_bit)-1
        if scale==0:
            scale=1/num
        elif scale<0:
            print('error scale when quantize.')
            return None
        out=torch.round(input/scale)
        out=nn.functional.hardtanh(out,-offset,num-offset)
        out = out * scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input=grad_output
        return grad_input, None, None,None
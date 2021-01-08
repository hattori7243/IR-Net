from torch.autograd import Function
import torch
import torch.nn as nn
import math


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * \
            (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class QuantizeNbit(Function):
    @staticmethod
    def forward(ctx, input,quan_bit=8):
        ctx.save_for_backward(input)
        num = math.pow(2,quan_bit)-1
        out = num*input
        torch.round_(out)
        out = out / num
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        return grad_input, None, None

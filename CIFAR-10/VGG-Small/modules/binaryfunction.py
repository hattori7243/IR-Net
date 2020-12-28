from torch.autograd import Function
import torch
import torch.nn as nn


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


class Quantize8bit(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = 127*input
        torch.round_(out)
        out = out / 127
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        return grad_input, None, None

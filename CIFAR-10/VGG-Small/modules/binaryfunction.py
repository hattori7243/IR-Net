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

'''
class my8BitQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros(input.shape).cuda()
        for i in range(0, out.shape[3], 8):
            out[:, :, :, i] = 128*input[:, :, :, i]
            out[:, :, :, i+1] = 64*input[:, :, :, i+1]
            out[:, :, :, i+2] = 32*input[:, :, :, i+2]
            out[:, :, :, i+3] = 16*input[:, :, :, i+3]
            out[:, :, :, i+4] = 8*input[:, :, :, i+4]
            out[:, :, :, i+5] = 4*input[:, :, :, i+5]
            out[:, :, :, i+6] = 2*input[:, :, :, i+6]
            out[:, :, :, i+7] = input[:, :, :, i+7]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        for i in range(0, grad_output.shape[2], 8):
            grad_input[:, :, :, i] = 128*grad_output[:, :, :, i]
            grad_input[:, :, :, i+1] = 64*grad_output[:, :, :, i+1]
            grad_input[:, :, :, i+2] = 32*grad_output[:, :, :, i+2]
            grad_input[:, :, :, i+3] = 16*grad_output[:, :, :, i+3]
            grad_input[:, :, :, i+4] = 8*grad_output[:, :, :, i+4]
            grad_input[:, :, :, i+5] = 4*grad_output[:, :, :, i+5]
            grad_input[:, :, :, i+6] = 2*grad_output[:, :, :, i+6]
            grad_input[:, :, :, i+7] = grad_output[:, :, :, i+7]
        grad_input = torch.clamp(grad_input, -1, +1)
        return grad_input, None, None
'''
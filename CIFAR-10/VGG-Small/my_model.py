from modules.ir_1w1a import Nomal_conv2d
from modules.ir_1w1a import Nomal_linear
from modules.ir_1w1a import IRConv2d
from modules import ir_1w1a
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 第一层归一化+8bit量化，其他层二值化
class VGG_SMALL_1W1A_normal(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A_normal, self).__init__()
        self.conv0 = Nomal_conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = IRConv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(
            128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = IRConv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = IRConv2d(
            256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = IRConv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = Nomal_linear(512*4*4, num_classes)
        #self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, Nomal_conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, Nomal_linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 每一层卷积层都归一化+8bit量化
class VGG_SMALL_allnormal(nn.Module):
    def __init__(self, num_classes=10,q_bit=8):
        super(VGG_SMALL_allnormal, self).__init__()
        self.conv0 = Nomal_conv2d(3, 128, kernel_size=3, padding=1, bias=False,quan=True,quan_bit=q_bit)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Nomal_conv2d(
            128, 128, kernel_size=3, padding=1, bias=False,quan=True,quan_bit=q_bit)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = Nomal_conv2d(
            128, 256, kernel_size=3, padding=1, bias=False,quan=True,quan_bit=q_bit)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = Nomal_conv2d(
            256, 256, kernel_size=3, padding=1, bias=False,quan=True,quan_bit=q_bit)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = Nomal_conv2d(
            256, 512, kernel_size=3, padding=1, bias=False,quan=True,quan_bit=q_bit)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = Nomal_conv2d(
            512, 512, kernel_size=3, padding=1, bias=False,quan=True,quan_bit=q_bit)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Nomal_conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VGG_SMALL_fullbit(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_fullbit, self).__init__()
        self.conv0 = nn.Conv2d(
            3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        #self.fc = nn.Linear(512*4*4, num_classes)
        self.fc=Nomal_linear(512*4*4,num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Nomal_conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




if __name__ == "__main__":

    model1=VGG_SMALL_1W1A_normal().cuda()
    model2=VGG_SMALL_fullbit().cuda()
    model3=VGG_SMALL_allnormal().cuda()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math
from modules import ir_1w1a

import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Calculate with ', device)

# Hyper parameters
batch_size = 256


# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
# 	transforms.RandomCrop(32, padding=4),
# 	transforms.RandomHorizontalFlip(),
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])


# transform_test = transforms.Compose([
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Load data

trainset = torchvision.datasets.CIFAR10(root='../../../data', train=True, download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../../data', train=False, download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(net, epoch=0):
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    print('\nEpoch: %d' % (epoch+1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print('=', end='')
    print('\nLoss: {:.3f} | Acc: {:.3f}%% ({}/{})'.
          format(train_loss, 100. * correct / total, correct, total))
    return 100. * correct / total


def test(net):
    global best_acc
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Test: Loss: {:.3f} | Acc: {:.3f}%% ({}/{})'.
              format(test_loss, 100. * correct / total, correct, total))
    return 100. * correct / total


class change_to_bi(nn.Module):
    def __init__(self, **kwargs):
        super(change_to_bi, self).__init__(**kwargs)

    def forward(self, x):
        origin = x*255
        new = torch.ones(x.shape[0], x.shape[1],
                         x.shape[2], x.shape[3]*8).to(device)
        for i in range(origin.shape[3]):
            new[:, :, :, 8*i+0] = origin[:, :, :, i]//128
            new[:, :, :, 8*i+1] = origin[:, :, :, i] % 128//64
            new[:, :, :, 8*i+2] = origin[:, :, :, i] % 64//32
            new[:, :, :, 8*i+3] = origin[:, :, :, i] % 32//16
            new[:, :, :, 8*i+4] = origin[:, :, :, i] % 16//8
            new[:, :, :, 8*i+5] = origin[:, :, :, i] % 8//4
            new[:, :, :, 8*i+6] = origin[:, :, :, i] % 4//2
            new[:, :, :, 8*i+7] = origin[:, :, :, i] % 2

        # 将0 1取值变成-1 1取值
        return (new-0.5)*2


class hold_to_bi(nn.Module):
    def __init__(self, width, height, inplanes=3, expansion=20):
        super(hold_to_bi, self).__init__()
        self.inplanes = inplanes
        self.expansion = expansion
        self.weight = nn.Parameter(torch.Tensor(
            self.inplanes*self.expansion, width, height)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = torch.zeros(
            x.shape[0], x.shape[1]*self.expansion, x.shape[2], x.shape[3]).to(device)
        x = x.repeat(1, self.expansion, 1, 1)
        for i in range(out.shape[0]):
            out[i, :, :, :] = x[i, :, :, :]-self.weight
        return out


class VGG_SMALL_bi_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_bi_1W1A, self).__init__()

        self.ch_bi = change_to_bi()
        #self.conv0 = ir_1w1a.my_Conv2d(3, 128, kernel_size=(
        #    3, 24), padding=(1, 8), stride=(1, 8), bias=False)
        self.conv0 = ir_1w1a.my_Conv2d(3, 128, kernel_size=(1,8), stride=(1,8),bias=False)

        # self.hold_to_bi = hold_to_bi(32, 32, inplanes=3, expansion=3)
        # self.conv0 = ir_1w1a.IRConv2d(
        #     9, 128, kernel_size=1, padding=0, bias=False)
        # self.conv0 = nn.Conv2d(3, 128, kernel_size=1, padding=0, bias=False)
        # self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        # self.conv0 = ir_1w1a.IRConv2d(
        #    3, 128, kernel_size = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(128)

        self.conv1 = ir_1w1a.IRConv2d(
            128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = ir_1w1a.IRConv2d(
            128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = ir_1w1a.IRConv2d(
            256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = ir_1w1a.IRConv2d(
            256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = ir_1w1a.IRConv2d(
            512, 512, kernel_size=3, padding=1, bias=False)
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
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.my_Conv2d):
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
        x = self.ch_bi(x)
        # x = self.hold_to_bi(x)
        x = self.conv0(x)
        # print(x)
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


model = VGG_SMALL_bi_1W1A().to(device)

T_min, T_max = 1e-1, 1e1
lr = 0.007
momentum = 0.9
weight_decay = 1e-4
epochs = 1000

best_acc = 0

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=lr,
#                       momentum = momentum, weight_decay = weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, epochs, eta_min=0, last_epoch=-1)


def Log_UP(K_min, K_max, epoch):
    Kmin, Kmax = math.log(
        K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / epochs * epoch)]).float().cuda()


for i in range(epochs):
    print('*'*128)
    t = Log_UP(T_min, T_max, i)
    if (t < 1):
        k = 1 / t
    else:
        k = torch.tensor([1]).float().cuda()
    print('k=', k.item(), ', t=', t.item())
    model.conv0.k = k
    model.conv1.k = k
    model.conv2.k = k
    model.conv3.k = k
    model.conv4.k = k
    model.conv5.k = k
    model.conv0.t = t
    model.conv1.t = t
    model.conv2.t = t
    model.conv3.t = t
    model.conv4.t = t
    model.conv5.t = t

    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train(model, i)
    t = test(model)
    if t > best_acc:
        best_acc = t
        torch.save(model.state_dict(), './model/ch_bi_best_adam.ckpt')
    print('best_acc=', best_acc)
    lr_scheduler.step()

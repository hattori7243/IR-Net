from vgg import VGG_SMALL_1W1A_modify
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
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# transform_train = transforms.Compose([
# 	transforms.RandomCrop(32, padding=4),
# 	transforms.RandomHorizontalFlip(),
# 	transforms.ToTensor(),
# ])


# transform_test = transforms.Compose([
# 	transforms.ToTensor(),
# ])

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
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
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
        print('-', end='')
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


model = VGG_SMALL_1W1A_modify().to(device)


def Log_UP(K_min, K_max, epoch, total_epoch):
    Kmin, Kmax = math.log(
        K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / total_epoch * epoch)]).float().cuda()


lr = 0.007
momentum = 0.9
weight_decay = 1e-4
total_epochs = 10

best_acc = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, total_epochs, eta_min=0, last_epoch=-1)

T_min, T_max = 1e-1, 1
epochs1 = 3
epochs2 = 2
epochs3 = 5

assert(total_epochs == epochs1+epochs2+epochs3)

for i in range(epochs1):
    print('*'*128)
    t = Log_UP(T_min, T_max, i, epochs1)
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
        #torch.save(model.state_dict(), './model/best_modify_1000.ckpt')
    print('best_acc=', best_acc)
    lr_scheduler.step()


for i in range(epochs1, epochs1+epochs2):
    print('*'*128)
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    print('k=', model.conv1.k.item(), ', t=', model.conv1.t.item())
    train(model, i)
    t = test(model)
    if t > best_acc:
        best_acc = t
        #torch.save(model.state_dict(), './model/best_modify_1000.ckpt')
    print('best_acc=', best_acc)
    lr_scheduler.step()

T_min, T_max = 1, 1000


for i in range(epochs1+epochs2, total_epochs):
    print('*'*128)
    t = Log_UP(T_min, T_max, i-epochs1-epochs2, epochs3)
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
        #torch.save(model.state_dict(), './model/best_modify_1000.ckpt')
    print('best_acc=', best_acc)
    lr_scheduler.step()

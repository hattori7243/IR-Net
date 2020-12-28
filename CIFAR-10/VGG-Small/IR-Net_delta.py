from vgg import VGG_SMALL_1W1A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math

import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Calculate with ', device)

# Hyper parameters
momentum = 0.9
weight_decay = 1e-4
batch_size = 256


criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

best_acc = 0


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
    print('\nTrain: Loss: {:.3f} | Acc: {:.3f}%% ({}/{})'.
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


model = VGG_SMALL_1W1A().to(device)
T_min, T_max = 1e-1, 1e1
lr = 0.007
epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, epochs, eta_min=0, last_epoch=-1)


def Log_UP(K_min, K_max, epoch):
    Kmin, Kmax = math.log(
        K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / epochs * epoch)]).float().cuda()



bast_acc = 0

conv1_weight_now=0
conv1_same_list=[]
conv2_weight_now=0
conv2_same_list=[]
conv3_weight_now=0
conv3_same_list=[]
conv4_weight_now=0
conv4_same_list=[]
conv5_weight_now=0
conv5_same_list=[]
acc_list=[]

for i in range(epochs):
    print('*'*128)
    t = Log_UP(T_min, T_max, i)
    if (t < 1):
        k = 1 / t
    else:
        k = torch.tensor([1]).float().cuda()
    print('k=', k.item(), ', t=', t.item())

    model.conv1.k = k
    model.conv2.k = k
    model.conv3.k = k
    model.conv4.k = k
    model.conv5.k = k

    model.conv1.t = t
    model.conv2.t = t
    model.conv3.t = t
    model.conv4.t = t
    model.conv5.t = t
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train(model, i)
    t = test(model)

    acc_list.append(t)

    if t > best_acc:
        best_acc = t
        torch.save(model.state_dict(), './model/delta_best.ckpt')
    print('best_acc=', best_acc)
    lr_scheduler.step()

    conv1_weight_pre=conv1_weight_now
    conv1_weight_now=model.conv1.bi_weight().cpu().detach().numpy().ravel()
    conv1_weight_product=conv1_weight_pre*conv1_weight_now
    conv1_same=0
    conv1_diff=0
    conv1_total=0
    for item in conv1_weight_product:
        if(item==1.0):
            conv1_same+=1
        else:
            conv1_diff+=1
        conv1_total+=1
    assert(conv1_same+conv1_diff==conv1_total)
    print('\n->conv1: total para={}, same={}%\n'.format(conv1_total,conv1_same/conv1_total*100))
    conv1_same_list.append(conv1_same/conv1_total*100)

    conv2_weight_pre=conv2_weight_now
    conv2_weight_now=model.conv2.bi_weight().cpu().detach().numpy().ravel()
    conv2_weight_product=conv2_weight_pre*conv2_weight_now
    conv2_same=0
    conv2_diff=0
    conv2_total=0
    for item in conv2_weight_product:
        if(item==1.0):
            conv2_same+=1
        else:
            conv2_diff+=1
        conv2_total+=1
    assert(conv2_same+conv2_diff==conv2_total)
    print('\n->conv2: total para={}, same={}%\n'.format(conv2_total,conv2_same/conv2_total*100))
    conv2_same_list.append(conv2_same/conv2_total*100)

    conv3_weight_pre=conv3_weight_now
    conv3_weight_now=model.conv3.bi_weight().cpu().detach().numpy().ravel()
    conv3_weight_product=conv3_weight_pre*conv3_weight_now
    conv3_same=0
    conv3_diff=0
    conv3_total=0
    for item in conv3_weight_product:
        if(item==1.0):
            conv3_same+=1
        else:
            conv3_diff+=1
        conv3_total+=1
    assert(conv3_same+conv3_diff==conv3_total)
    print('\n->conv3: total para={}, same={}%\n'.format(conv3_total,conv3_same/conv3_total*100))
    conv3_same_list.append(conv3_same/conv3_total*100)

    conv4_weight_pre=conv4_weight_now
    conv4_weight_now=model.conv4.bi_weight().cpu().detach().numpy().ravel()
    conv4_weight_product=conv4_weight_pre*conv4_weight_now
    conv4_same=0
    conv4_diff=0
    conv4_total=0
    for item in conv4_weight_product:
        if(item==1.0):
            conv4_same+=1
        else:
            conv4_diff+=1
        conv4_total+=1
    assert(conv4_same+conv4_diff==conv4_total)
    print('\n->conv4: total para={}, same={}%\n'.format(conv4_total,conv4_same/conv4_total*100))
    conv4_same_list.append(conv4_same/conv4_total*100)

    conv5_weight_pre=conv5_weight_now
    conv5_weight_now=model.conv5.bi_weight().cpu().detach().numpy().ravel()
    conv5_weight_product=conv5_weight_pre*conv5_weight_now
    conv5_same=0
    conv5_diff=0
    conv5_total=0
    for item in conv5_weight_product:
        if(item==1.0):
            conv5_same+=1
        else:
            conv5_diff+=1
        conv5_total+=1
    assert(conv5_same+conv5_diff==conv5_total)
    print('\n->conv5: total para={}, same={}%\n'.format(conv5_total,conv5_same/conv5_total*100))
    conv5_same_list.append(conv5_same/conv5_total*100)

    with open('./same/conv1.out','w') as f:
        for item in conv1_same_list:
            f.write(str(item))
            f.write(',')
    with open('./same/conv2.out','w') as f:
        for item in conv2_same_list:
            f.write(str(item))
            f.write(',')
    with open('./same/conv3.out','w') as f:
        for item in conv3_same_list:
            f.write(str(item))
            f.write(',')
    with open('./same/conv4.out','w') as f:
        for item in conv4_same_list:
            f.write(str(item))
            f.write(',')
    with open('./same/conv5.out','w') as f:
        for item in conv5_same_list:
            f.write(str(item))
            f.write(',')
    with open('./same/acc.out','w') as f:
        for item in acc_list:
            f.write(str(item))
            f.write(',')

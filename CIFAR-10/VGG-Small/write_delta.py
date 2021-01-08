from tqdm import tqdm
from modules.ir_1w1a import Nomal_conv2d
from modules.ir_1w1a import Nomal_linear
from modules import ir_1w1a
from vgg import VGG_SMALL_1W1A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math
import numpy
import os
import my_model
import struct


####################################################################################
####################################################################################
####################################################################################

####################################################################################
## 将二进制的权重的list组合成uint8的list，便于写入文件
## 将二进制的权重的list组合成uint8的list，便于写入文件
def bi_list_To_uint8_list(bi_list,n_1=1.0,n_0=-1.0):
    if type(bi_list)==numpy.ndarray:
        bi_list=bi_list.tolist()
    lenth=len(bi_list)

    # 对齐到字节
    if lenth%8!=0:
        for i in range(8-(lenth%8)):
            bi_list.append(n_0)
        lenth=lenth+8-lenth%8

    uint8_list=[]

    for i in range(lenth):
        if bi_list[i]==n_1:
            bi_list[i]=1
        elif bi_list[i]==n_0:
            bi_list[i]=0
        elif bi_list[i]==0.0:
            bi_list[i]=1
        else:
            print('error',bi_list[i])
            return

    for i in range(0,lenth,8):
        uint8_i=bi_list[i]*128+bi_list[i+1]*64+bi_list[i+2]*32+bi_list[i+3]*16+bi_list[i+4]*8+bi_list[i+5]*4+bi_list[i+6]*2+bi_list[i+7]
        uint8_i=int(uint8_i)
        uint8_list.append(uint8_i)
    return uint8_list

####################################################################################
model_path='./out/partnormal88.97/'
model = my_model.VGG_SMALL_1W1A_normal().cuda()

for i in tqdm(range(250)):

    ####################################################################################
    ### 保存两个版本的参数

    model.load_state_dict(torch.load(model_path+str(i)+'.ckpt'))
    first0=numpy.round(127*model.conv0.true_weight().detach().cpu().numpy().ravel()).astype(int)
    ## 1-5为二值化层
    first1=model.conv1.true_weight().detach().cpu().numpy().ravel()
    first2=model.conv2.true_weight().detach().cpu().numpy().ravel()
    first3=model.conv3.true_weight().detach().cpu().numpy().ravel()
    first4=model.conv4.true_weight().detach().cpu().numpy().ravel()
    first5=model.conv5.true_weight().detach().cpu().numpy().ravel()


    first_fc_weight=numpy.round(127*model.fc.true_weight()[0].detach().cpu().numpy().ravel()).astype(int)
    first_fc_bias=numpy.round(127*model.fc.true_weight()[1].detach().cpu().numpy().ravel()).astype(int)

    model.load_state_dict(torch.load(model_path+str(i+1)+'.ckpt'))
    second0=numpy.round(127*model.conv0.true_weight().detach().cpu().numpy().ravel()).astype(int)
    second1=model.conv1.true_weight().detach().cpu().numpy().ravel()
    ## 1-5为二值化层
    second1=model.conv1.true_weight().detach().cpu().numpy().ravel()
    second2=model.conv2.true_weight().detach().cpu().numpy().ravel()
    second3=model.conv3.true_weight().detach().cpu().numpy().ravel()
    second4=model.conv4.true_weight().detach().cpu().numpy().ravel()
    second5=model.conv5.true_weight().detach().cpu().numpy().ravel()
    second_fc_weight=numpy.round(127*model.fc.true_weight()[0].detach().cpu().numpy().ravel()).astype(int)
    second_fc_bias=numpy.round(127*model.fc.true_weight()[1].detach().cpu().numpy().ravel()).astype(int)
    ####################################################################################
    ## 把两个版本的差量写到文件中

    out_path='./part_diff/'+str(i)+'.out'
    ##清理文件
    with open(out_path,'w') as binfile:
        print('clean the ',out_path)
    ##写入
    with open(out_path,'ab') as binfile:
        #################
        for x,y in zip(first0,second0):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))
        #################
        xnor_diff=(first1*second1)
        xnor_diff=bi_list_To_uint8_list(xnor_diff)
        for t in xnor_diff:
            binfile.write(struct.pack('B',t))
        #################
        xnor_diff=(first2*second2)
        xnor_diff=bi_list_To_uint8_list(xnor_diff)
        for t in xnor_diff:
            binfile.write(struct.pack('B',t))
        #################
        xnor_diff=(first3*second3)
        xnor_diff=bi_list_To_uint8_list(xnor_diff)
        for t in xnor_diff:
            binfile.write(struct.pack('B',t))
        #################
        xnor_diff=(first4*second4)
        xnor_diff=bi_list_To_uint8_list(xnor_diff)
        for t in xnor_diff:
            binfile.write(struct.pack('B',t))
        #################
        xnor_diff=(first5*second5)
        xnor_diff=bi_list_To_uint8_list(xnor_diff)
        for t in xnor_diff:
            binfile.write(struct.pack('B',t))
        #################
        for x,y in zip(first_fc_weight,second_fc_weight):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))
        #################
        for x,y in zip(first_fc_bias,second_fc_bias):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

####################################################################################
####################################################################################
####################################################################################
'''
def file_clean(path):
    with open(path,'w') as file:
        pass
    print('clean the ',path)
    return

model_path='./out/allnormal89.33/'

model = my_model.VGG_SMALL_1W1A_normal().cuda()

for i in tqdm(range(999)):
#for i in (range(999)):
    model.load_state_dict(torch.load(model_path+str(i)+'.ckpt'))
    first0=numpy.round(127*model.conv0.true_weight().detach().cpu().numpy().ravel()).astype(int)
    first1=numpy.round(127*model.conv1.true_weight().detach().cpu().numpy().ravel()).astype(int)
    first2=numpy.round(127*model.conv2.true_weight().detach().cpu().numpy().ravel()).astype(int)
    first3=numpy.round(127*model.conv3.true_weight().detach().cpu().numpy().ravel()).astype(int)
    first4=numpy.round(127*model.conv4.true_weight().detach().cpu().numpy().ravel()).astype(int)
    first5=numpy.round(127*model.conv5.true_weight().detach().cpu().numpy().ravel()).astype(int)
    first_fc_weight=numpy.round(127*model.fc.true_weight()[0].detach().cpu().numpy().ravel()).astype(int)
    first_fc_bias=numpy.round(127*model.fc.true_weight()[1].detach().cpu().numpy().ravel()).astype(int)

    model.load_state_dict(torch.load(model_path+str(i+1)+'.ckpt'))
    second0=numpy.round(127*model.conv0.true_weight().detach().cpu().numpy().ravel()).astype(int)
    second1=numpy.round(127*model.conv1.true_weight().detach().cpu().numpy().ravel()).astype(int)
    second2=numpy.round(127*model.conv2.true_weight().detach().cpu().numpy().ravel()).astype(int)
    second3=numpy.round(127*model.conv3.true_weight().detach().cpu().numpy().ravel()).astype(int)
    second4=numpy.round(127*model.conv4.true_weight().detach().cpu().numpy().ravel()).astype(int)
    second5=numpy.round(127*model.conv5.true_weight().detach().cpu().numpy().ravel()).astype(int)
    second_fc_weight=numpy.round(127*model.fc.true_weight()[0].detach().cpu().numpy().ravel()).astype(int)
    second_fc_bias=numpy.round(127*model.fc.true_weight()[0].detach().cpu().numpy().ravel()).astype(int)


    out_path='./diff_conv/'+str(i)+'.out'
    file_clean(out_path)

    with open(out_path,'ab') as binfile:
####################################################################################
####################################################################################
# 两个[-128,+127]的数的差值范围为[-255,+255]，远远超过了int8所能表示的范围
# 所以不能直接对两个做差得到差量
# 将[-128,+127]变成一个有向循环的区间，差量的取值范围为[0,255]
# 将差量表示成为下一个版本减上一个版本的值，若值大于0，则直接存储，若值小于0，则用256+结果的值存储（类似反码）
# 在解压时，当前版本加上差量，若结果在[-128,+127]之间，则结束
# 若结果>+127，则将结果减去256（即将+128连接到-128），则可得到正确结果
# 
# 例如:+100->-100,差量=-200(小于0)，则保存-200+256=56
# 恢复时，下一版本=+100+56=156>127，则减去256=156-256=100
# 
####################################################################################
####################################################################################
        for x,y in zip(first0,second0):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

        for x,y in zip(first1,second1):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

        for x,y in zip(first2,second2):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

        for x,y in zip(first3,second3):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

        for x,y in zip(first4,second4):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

        for x,y in zip(first5,second5):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

        for x,y in zip(first_fc_weight,second_fc_weight):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))

        for x,y in zip(first_fc_bias,second_fc_bias):
            t=y-x
            if t<0:
                t=t+256
            binfile.write(struct.pack('B',t))
            
    print('success write the ',out_path)
'''
####################################################################################
####################################################################################
####################################################################################
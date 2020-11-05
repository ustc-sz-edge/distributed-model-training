import torch.nn as nn
import torch
import re

def MakeLayers(params_list):
    conv_layers = nn.Sequential()
    fc_layers = nn.Sequential()
    c_idx=p_idx=f_idx=1
    for param in params_list:
        if param[1] == 'Conv':
            conv_layers.add_module(param[0], nn.Conv2d(
                param[2][0], param[2][1], param[2][2], param[2][3],param[2][4]))
            if len(param) >=4:
                if param[3] == 'Batchnorm':
                    conv_layers.add_module(
                        'batchnorm'+str(c_idx), nn.BatchNorm2d(param[2][1]))
                if param[3]=='Relu' or (param[3] == 'Batchnorm' and param[4]=='Relu'):
                    conv_layers.add_module('relu'+str(c_idx),nn.ReLU(inplace=True))
                else:
                    conv_layers.add_module('sigmoid'+str(c_idx),nn.Sigmoid())
            if len(param) >=6 :
                if param[3] == 'Batchnorm':
                    if param[5]=='Maxpool':
                        conv_layers.add_module(
                            'maxpool'+str(p_idx), nn.MaxPool2d(param[6][0], param[6][1],param[6][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool'+str(p_idx), nn.AvgPool2dparam[6][0], param[6][1],param[6][2])
                else:
                    if param[4]=='Maxpool':
                        conv_layers.add_module(
                            'maxpool'+str(p_idx), nn.MaxPool2d(param[5][0], param[5][1],param[5][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool'+str(p_idx), nn.AvgPool2dparam[5][0], param[5][1],param[5][2])
                p_idx+=1
            c_idx+=1
            
        else:
            fc_layers.add_module(param[0], nn.Linear(param[2][0], param[2][1]))
            if len(param) >= 4:
                if param[3] == 'Dropout':
                    fc_layers.add_module('dropout', nn.Dropout(param[4]))
                if param[3] == 'Relu' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Relu'):
                    fc_layers.add_module('relu'+str(f_idx), nn.ReLU(inplace=True))
                elif param[3] == 'Sigmoid' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Sigmoid'):
                    fc_layers.add_module('sigmoid'+str(f_idx), nn.Sigmoid())
                elif param[3] == 'Softmax' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Softmax'):
                    fc_layers.add_module('softmax'+str(f_idx), nn.Softmax())
            f_idx+= 1
    return conv_layers, fc_layers


class MyNet(nn.Module):
    def __init__(self, params_list):
        super(MyNet, self).__init__()
        self.features, self.classifier = MakeLayers(params_list)

    def forward(self, x):
        if len(self.features) > 0:
            feature = self.features(x)
            linear_input = torch.flatten(feature, 1)
            output = self.classifier(linear_input)
        else:
            output = self.classifier(x)
        return output

#初始化模型的参数由以下几部分构成：
# 卷积层(每一层的名字,Conv,参数列表( , , , , ),Batchnorm(可选),激活函数(Relu,Sigmoid)，池化层(Maxpool,Avgpool,可选),参数列表( , , ))
#全连接层(每一层的名字,FC,参数列表( , ),Dropout(可选）,激活函数(Relu,Sigmoid,Softmax,可选))


LeNet5 = [('conv1', 'Conv', (1, 6, 5, 1,1), 'Sigmoid', 'Maxpool',(2,2,0)), ('conv2', 'Conv', (6, 15, 5, 1,0),'Sigmoid', 'Maxpool',(2,2,0)), 
             ('fc1', 'FC', (16*4*4, 120), 'Dropout',0.2,'Sigmoid'), ('fc2', 'FC', (120, 84), 'Sigmoid'),('fc3', 'FC', (84, 10))]


AlexNet=[('conv1', 'Conv', (3, 64, 11, 4,2), 'Relu', 'Maxpool',(3,2,0)), ('conv2', 'Conv', (64, 192, 5, 1,2),'Relu', 'Maxpool',(3,2,0)), 
            ('conv3', 'Conv', (192, 384, 3, 1,1), 'Relu'), ('conv4', 'Conv', (384, 256, 3, 1,1),'Relu'),
            ('conv5', 'Conv', (256, 256, 3, 1,1), 'Relu', 'Maxpool',(3,2,0)),
            ('fc1', 'FC', (256*6*6, 4096), 'Relu'), ('fc2', 'FC', (4096, 4096), 'Relu'),('fc3', 'FC', (4096, 1000), 'Relu')]

VGG16=[('conv1_1', 'Conv', (3, 64, 3, 1,0), 'Batchnorm','Relu'), ('conv1_2', 'Conv', (64, 64, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv2_1', 'Conv', (64, 128, 3, 1,0), 'Relu'), ('conv2_2', 'Conv', (128, 128, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv3_1', 'Conv', (128, 256, 3, 1,0), 'Relu'), ('conv3_2', 'Conv', (256, 256, 3, 1,1),'Relu'), ('conv3_3', 'Conv', (256, 256, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv4_1', 'Conv', (256, 512, 3, 1,0), 'Relu'), ('conv4_2', 'Conv', (512, 512, 3, 1,1),'Relu'), ('conv4_3', 'Conv', (512, 512, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('conv5_1', 'Conv', (512, 512, 3, 1,0), 'Relu'), ('conv5_2', 'Conv', (512, 512, 3, 1,1),'Relu'), ('conv5_3', 'Conv', (512, 512, 3, 1,1),'Relu', 'Maxpool',(2,2,1)),
        ('fc1', 'FC', (512*7*7, 4096), 'Relu'), ('fc2', 'FC', (4096, 4096), 'Relu'),('fc3', 'FC', (4096, 1000))]
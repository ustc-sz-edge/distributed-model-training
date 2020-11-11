import torch.nn as nn
import torch
import re
import sys
import time
import numpy as np
import torch.nn.functional as F
import gc
from utils import printer, time_since

# def MakeLayers(params_list):
#     conv_layers = nn.Sequential()
#     fc_layers = nn.Sequential()
#     c_idx=p_idx=f_idx=1
#     for param in params_list:
#         if param[1] == 'Conv':
#             conv_layers.add_module(param[0], nn.Conv2d(
#                 param[2][0], param[2][1], param[2][2], param[2][3],param[2][4]))
#             if len(param) >=4:
#                 if param[3] == 'Batchnorm':
#                     conv_layers.add_module(
#                         'batchnorm'+str(c_idx), nn.BatchNorm2d(param[2][1]))
#                 if param[3]=='Relu' or (param[3] == 'Batchnorm' and param[4]=='Relu'):
#                     conv_layers.add_module('relu'+str(c_idx),nn.ReLU(inplace=True))
#                 else:
#                     conv_layers.add_module('sigmoid'+str(c_idx),nn.Sigmoid())
#             if len(param) >=6 :
#                 if param[3] == 'Batchnorm':
#                     if param[5]=='Maxpool':
#                         conv_layers.add_module(
#                             'maxpool'+str(p_idx), nn.MaxPool2d(param[6][0], param[6][1],param[6][2]))
#                     else:
#                         conv_layers.add_module(
#                             'avgpool'+str(p_idx), nn.AvgPool2dparam[6][0], param[6][1],param[6][2])
#                 else:
#                     if param[4]=='Maxpool':
#                         conv_layers.add_module(
#                             'maxpool'+str(p_idx), nn.MaxPool2d(param[5][0], param[5][1],param[5][2]))
#                     else:
#                         conv_layers.add_module(
#                             'avgpool'+str(p_idx), nn.AvgPool2dparam[5][0], param[5][1],param[5][2])
#                 p_idx+=1
#             c_idx+=1
            
#         else:
#             fc_layers.add_module(param[0], nn.Linear(param[2][0], param[2][1]))
#             if len(param) >= 4:
#                 if param[3] == 'Dropout':
#                     fc_layers.add_module('dropout', nn.Dropout(param[4]))
#                 if param[3] == 'Relu' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Relu'):
#                     fc_layers.add_module('relu'+str(f_idx), nn.ReLU(inplace=True))
#                 elif param[3] == 'Sigmoid' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Sigmoid'):
#                     fc_layers.add_module('sigmoid'+str(f_idx), nn.Sigmoid())
#                 elif param[3] == 'Softmax' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Softmax'):
#                     fc_layers.add_module('softmax'+str(f_idx), nn.Softmax())
#             f_idx+= 1
#     return conv_layers, fc_layers


# class MyNet(nn.Module):
#     def __init__(self, params_list):
#         super(MyNet, self).__init__()
#         self.features, self.classifier = MakeLayers(params_list)

#     def forward(self, x):
#         if len(self.features) > 0:
#             feature = self.features(x)
#             linear_input = torch.flatten(feature, 1)
#             output = self.classifier(linear_input)
#         else:
#             output = self.classifier(x)
#         return F.log_softmax(output, dim=1)

class MyNet(nn.Module):
    def __init__(self, params_list):
        super(MyNet, self).__init__()
        self.params_list = params_list
        for param in params_list:
            # layers = []
            if param[1] == 'Conv':
                layers = nn.Conv2d(
                    param[2][0], param[2][1], param[2][2], param[2][3],param[2][4])
                if len(param) >=4:
                    if param[3] == 'Batchnorm':
                        layers.append(nn.BatchNorm2d(param[2][1]))
                    if param[3]=='Relu' or (param[3] == 'Batchnorm' and param[4]=='Relu'):
                        layers.append(nn.ReLU(inplace=True))
                    else:
                        layers.append(nn.Sigmoid())
                if len(param) >=6 :
                    if param[3] == 'Batchnorm':
                        if param[5]=='Maxpool':
                            layers.append(nn.MaxPool2d(param[6][0], param[6][1],param[6][2]))
                        else:
                            layers.append(nn.AvgPool2dparam[6][0], param[6][1],param[6][2])
                    else:
                        if param[4]=='Maxpool':
                            layers.append(nn.MaxPool2d(param[5][0], param[5][1],param[5][2]))
                        else:
                            layers.append(nn.AvgPool2dparam[5][0], param[5][1],param[5][2])
            else:
                layers = nn.Linear(param[2][0], param[2][1])
                if len(param) >= 4:
                    if param[3] == 'Dropout':
                        layers.append(nn.Dropout(param[4]))
                    if param[3] == 'Relu' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Relu'):
                        layers.append(nn.ReLU(inplace=True))
                    elif param[3] == 'Sigmoid' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Sigmoid'):
                        layers.append(nn.Sigmoid())
                    elif param[3] == 'Softmax' or (param[3]=='Dropout' and len(param)==6 and param[5]=='Softmax'):
                        layers.append(nn.Softmax())
            setattr(self, param[0], layers)

    def forward(self, x):
        for param in self.params_list:
            layer = getattr(self, param[0])
            x = layer(x)
        return F.log_softmax(x, dim=1)



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


def train(args, config, tx2_model, device, tx2_train_loader, tx2_test_loader, tx2_optimizer, epoch, fid):

    vm_start = time.time()
    tx2_model.train()

    for li_idx in range(args.local_iters):

        for batch_idx, (vm_data, vm_target) in enumerate(tx2_train_loader, 1):
            # print(data.location)
            # print("vm data:", vm_data)

            if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                if args.model_type == 'LR':
                    vm_data = vm_data.squeeze(1) 
                    vm_data = vm_data.view(-1, 28 * 28)
                else:
                    pass

            if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                if args.model_type == 'LSTM':
                    vm_data = vm_data.permute(0, 2, 3, 1)
                    vm_data = vm_data.contiguous().view(-1, 32, 32 * 3)                    
                else:
                    pass
                
            vm_data, vm_target = vm_data.to(device), vm_target.to(device)
            # print('--[Debug] vm_data = ', vm_data.get())
            if args.model_type == 'LSTM':
                hidden = tx2_model.initHidden(args.batch_size)
                hidden = hidden.send(vm_data.location)
                for col_idx in range(32):
                    vm_data_col = vm_data[:, col_idx, :]
                    vm_output, hidden = tx2_model(vm_data_col, hidden)
            else:
                vm_output = tx2_model(vm_data)

            tx2_optimizer.zero_grad()
            
            vm_loss = F.nll_loss(vm_output, vm_target)
            vm_loss.backward()
            tx2_optimizer.step()
            # vm_data = vm_data.get()

            if batch_idx % args.log_interval == 0:
                vm_loss = vm_loss.item()  # <-- NEW: get the loss back
                #print("Epoch :{} batch_idx: {} print ok".format(epoch, batch_idx))
                # print(vm_loss)
                printer('-->[{}] Train Epoch: {} Local Iter: {} tx2: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time_since(vm_start), epoch, li_idx, config.idx, batch_idx * args.batch_size, 
                    len(tx2_train_loader) * args.batch_size,
                    100. * batch_idx / len(tx2_train_loader), vm_loss), fid)

            # vm_data = vm_data.get()
            # vm_target = vm_target.get()
            # vm_output = vm_output.get()

            # if not batch_idx % args.log_interval == 0:
            #     vm_loss = vm_loss.get()
            
            del vm_data
            del vm_target
            del vm_output
            del vm_loss


    if args.enable_vm_test:
        printer('-->[{}] Test set: Epoch: {} tx2: {}'.format(time_since(vm_start), epoch, config.idx), fid)
        # <--test for each vm
        test(args, vm_start, tx2_model, device, tx2_test_loader, epoch, fid)

        # vm_models[vm_idx].move(param_server)
        # vm_models[vm_idx] = vm_models[vm_idx].get()
        # torch.cuda.empty_cache()
        gc.collect()

def test(args, start, model, device, test_loader, epoch, fid):
    model.eval()

    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                if args.model_type == 'LR':
                    data = data.squeeze(1) 
                    data = data.view(-1, 28 * 28)
                else:
                    pass

            if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                if args.model_type == 'LSTM':
                    data = data.view(-1, 32, 32 * 3)                    
                else:
                    pass  

            if args.model_type == 'LSTM':
                hidden = model.initHidden(args.test_batch_size)
                hidden = hidden.send(data.location)
                for col_idx in range(32):
                    data_col = data[:, col_idx, :]
                    output, hidden = model(data_col, hidden)
            else:
                output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            #print('--[Debug][in Test set] batch correct:', batch_correct)

            # if not args.enable_vm_test:
            #     printer('--[Debug][in Test set] batch correct: {}'.format(batch_correct), fid)
            
            # data =  data.get()
            # target = target.get()
            # output = output.get()
            # pred = pred.get()
            if args.model_type == 'LSTM':
                #hidden = hidden.get()
                del hidden
                
            del data
            del target
            del output
            del pred
            del batch_correct

    test_loss /= len(test_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(test_loader.dataset))

    if args.enable_vm_test:  
        printer('-->[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy), fid)
    else:
        printer('[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy), fid)

    gc.collect()

    return test_loss, test_accuracy
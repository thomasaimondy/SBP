# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "train.py" - Initializing the network, optimizer and loss for training and testing.

 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import models
from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import sys
from torch import set_default_tensor_type

cossim = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)

def train(args, device, train_loader, traintest_loader, test_loader):
    torch.manual_seed(args.seed) #42, 10, 109, 23, 87

    for trial in range(1,args.trials+1):
        # Network topology

        model = models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, spike_window=args.spike_window,  device=device, thresh=args.thresh, randKill=args.randKill, lens=args.lens, decay=args.decay, xishuspike=args.xishuspike, xishuada=args.xishuada, propnoise=args.propnoise, dataset=args.dataset)

        if args.cuda:
            model.cuda()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)#weight_decay是L2权重惩罚，原有程序默认为0，使用BiRNN的设置值为1e-5
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lrstep, gamma=args.lrgama)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")

        # Loss function
        print(model)
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        # elif args.loss == 'CE':
        #     loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        elif args.loss == 'CE':
            # loss = (nn.CrossEntropyLoss(), (lambda L : L.reshape(L.numel(),)))
            loss = (nn.CrossEntropyLoss(), (lambda l : l))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,))

        filepath = 'model/' + args.codename.split('-')[0] + '/' + args.codename
        if os.path.exists(filepath+'/model.pth') and args.cont=='True':
            checkpoint = torch.load(filepath+'/model.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            #scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']+1
            print('加载 epoch {} 成功！继续从epoch {} 开始训练。'.format(start_epoch-1,start_epoch))
        else:
            start_epoch = 1
            print('无保存模型，将从头开始训练！')

        for epoch in range(start_epoch, args.epochs + 1):
            # Training

            train_epoch(args, model, device, train_loader, optimizer, loss, onoffnoise=args.train_noise)
            scheduler.step()

            # Compute accuracy on training and testing set
            print("\nSummary of epoch %d:" % (epoch))
            test_epoch(args, model, device, traintest_loader, loss, 'Train',epoch, onoffnoise=args.test_noise)
            test_epoch(args, model, device, test_loader, loss, 'Test',epoch, onoffnoise=args.test_noise)
            if args.cont=='True':
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                #state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}
                torch.save(state, filepath+'/model.pth')


def train_epoch(args, model, device, train_loader, optimizer, loss, onoffnoise):
    model.train()

    if args.freeze_conv_layers:
        for i in range(model.conv_to_fc):
            for param in model.layers[i].conv.parameters():
                param.requires_grad = False
        for param in model.layers[1].fc.parameters():
            param.requires_grad = False
        for param in model.layers[1].rec.parameters():
            param.requires_grad = False

    if args.dataset == 'dvsgesture':
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            data = data.squeeze(1).view(data.shape[0],-1,data.shape[-1]*data.shape[-2])
            label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)    # add 0929
            label = label.unsqueeze(1).repeat(1,args.spike_window,1)

            max_len = data.shape[-2]
            optimizer.zero_grad()
            output = model(data, label, max_len, onoffnoise, args.dataset) # cx：后面的loss计算、backward、step什么的你自己补上
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()

    if args.dataset == 'tidigits' or args.dataset=='MNIST':
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            data = data.squeeze(1)
            label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)    # add 0929
            label = label.unsqueeze(1).repeat(1,args.spike_window,1)

            max_len = data.shape[-1]
            optimizer.zero_grad()
            output = model(data, label, max_len, onoffnoise, args.dataset) # cx：后面的loss计算、backward、step什么的你自己补上
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()
    if args.dataset=='nettalk':
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            data = data.squeeze(1)
            label = label.float().unsqueeze(1).repeat(1,args.spike_window,1)
            max_len = data.shape[-1]
            optimizer.zero_grad()
            output = model(data, label, max_len, onoffnoise, args.dataset) # cx：后面的loss计算、backward、step什么的你自己补上
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()
    if args.dataset == 'timit':
        for batch_idx, (data, label, seq_len) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            # timit data set sequencial process
            #     data = data.unfold(1,args.spike_window,1) # {batch_size}{samples}{binds}{time_window}
            #     data = data.permute(0,1,3,2) # {batch_size}{samples}{time_window}{binds}
            #     label = label.unfold(1,args.spike_window,1) # {batch_size}{samples}{binds}{time_window}
            #     targets = torch.zeros(label.shape[0], label.shape[1], args.spike_window, args.label_features, device=device).scatter_(3, label.unsqueeze(3).long(), 1.0)    # add 0929
            #     optimizer.zero_grad()
            #     for ii in range(data.shape[1]):
            #         onesample = data[:,ii,:,:]
            #         onetarget = targets[:,ii,:,:]
            #         output = model(onesample, onetarget)
            #         loss_val = loss[0](output, loss[1](onetarget))
            #         loss_val.backward(retain_graph = True)
            #         optimizer.step()
            max_len = seq_len.max()
            data = data[:, :max_len, :]
            label = label[:, :max_len] # cx：尽量截短补零的长度，batch size内对齐

            # label = label.unfold(1,args.spike_window,1) # {batch_size}{frames}{39}
            label = torch.zeros(label.shape[0], label.shape[1], args.label_features, device=device).scatter_(2, label.unsqueeze(2).long(), 1.0)    # add 0929

            optimizer.zero_grad()
            output = model(data, label, max_len, onoffnoise,args.dataset) # cx：后面的loss计算、backward、step什么的你自己补上
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()

    if args.dataset == 'others':
        for batch_idx, (data, label, seq_len) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)
            optimizer.zero_grad()
            output = model(data, targets)
            loss_val = loss[0](output, loss[1](targets))
            loss_val.backward(retain_graph = True)
            optimizer.step()

def writefile(args, file):
    filepath = 'output/'+args.codename.split('-')[0]+'/'+args.codename
    filetestloss = open(filepath + file, 'a')
    return filetestloss

def test_epoch(args, model, device, test_loader, loss, phase, epoch, onoffnoise):
    model.eval()

    test_loss, correct = 0, 0
    total = 0

    with torch.no_grad():
        if args.dataset == 'dvsgesture':
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                data = data.squeeze(1).view(data.shape[0],-1,data.shape[-1]*data.shape[-2])
                label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(
                    1).long(), 1.0)  # add 0929
                label = label.unsqueeze(1).repeat(1, args.spike_window, 1)

                max_len = data.shape[-2]

                output = model(data, None, max_len, onoffnoise,args.dataset)

                test_loss += loss[0](output, loss[1](label)).item()
                phn_prediction = torch.argmax(output, dim=2)
                targets = torch.argmax(label, dim=2)

                mach1, number1 = (targets == phn_prediction).float().sum(), targets.numel()
                correct += (mach1 / number1).item()

        if args.dataset == 'tidigits' or args.dataset=='MNIST':
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                data = data.squeeze(1)
                label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(
                    1).long(), 1.0)  # add 0929
                label = label.unsqueeze(1).repeat(1, args.spike_window, 1)

                max_len = data.shape[-1]

                output = model(data, None, max_len, onoffnoise,args.dataset)

                test_loss += loss[0](output, loss[1](label)).item()
                phn_prediction = torch.argmax(output, dim=2)
                targets = torch.argmax(label, dim=2)

                mach1, number1 = (targets == phn_prediction).float().sum(), targets.numel()
                correct += (mach1 / number1).item()

        if args.dataset == 'nettalk':
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                data = data.squeeze(1)
                labels = label.float().unsqueeze(1).repeat(1, args.spike_window, 1)

                max_len = data.shape[-1]
                outputs = model(data, None, max_len, onoffnoise,args.dataset)
                total += 1
                for i in range(args.spike_window):
                    output = outputs[:,i,:]
                    label = labels[:, i, :]
                    if output.max() >= 0.05:
                        pos = []
                        for label_i in range(26):
                            if (label[0, label_i] != 0) or (output[0, label_i] != 0):
                                pos.append(label_i)
                        tem_out = torch.zeros((1, len(pos)))
                        tem_lab = torch.zeros((1, len(pos)))
                        for label_i in range(len(pos)):
                            tem_out[0, label_i] = output[0, pos[label_i]]
                            tem_lab[0, label_i] = label[0, pos[label_i]]
                        correct += cossim(tem_out, tem_lab)/args.spike_window
                    else:
                        correct += 0

        if args.dataset == 'timit':
            for batch_idx, (data, label, seq_len) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                max_len = seq_len.max()
                data = data[:, :max_len, :]
                label = label[:, :max_len]  # cx：尽量截短补零的长度，batch size内对齐

                # label = label.unfold(1,args.spike_window,1) # {batch_size}{frames}{39}
                label = torch.zeros(label.shape[0], label.shape[1], args.label_features, device=device).scatter_(2, label.unsqueeze(2).long(),1.0)  # add 0929


                output = model(data, None, max_len, onoffnoise,args.dataset)

                test_loss += loss[0](output, loss[1](label)).item()
                phn_prediction = torch.argmax(output, dim=2)
                targets = torch.argmax(label,dim=2)

                mach1, number1 = 0, 0
                for ic in range(phn_prediction.shape[0]):
                    mach1 += (phn_prediction[ic,:seq_len[ic]] == targets[ic,:seq_len[ic]]).float().sum()
                    number1 += phn_prediction[ic,:seq_len[ic]].numel()

                #print(phn_prediction.shape,phn_prediction[0,:])
                #print(label.shape,label[0,0:])
                #correct += ((output == label).float()).sum() / label.numel()
                correct += (mach1/number1).item()

        if args.dataset == 'others':
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1,label.unsqueeze(1).long(), 1.0)

            output = model(data, None)

            test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred).long()).sum().item()
            # print(((phn_prediction == targets).float()).sum() / targets.numel(),correct)

    loss = test_loss / (batch_idx+1)
    if True:#not args.regression:
        if args.dataset == 'nettalk':
            acc = 100. * correct / total
        else:
            acc = 100. * correct / (batch_idx+1)
        print("\t[%5sing set] Loss: %6f, Accuracy: %6.2f%%" % (phase, loss, acc))

        filetestloss = writefile(args, '/testloss.txt')
        filetestacc = writefile(args, '/testacc.txt')
        filetrainloss = writefile(args, '/trainloss.txt')
        filetrainacc = writefile(args, '/trainacc.txt')

        if phase == 'Train':
            #writer.add_scalar('train_loss', loss, epoch)
            #writer.add_scalar('train_acc', acc, epoch)
            filetrainloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetrainacc.write(str(epoch) + ' ' + str(acc) + '\n')
            filetrainloss.close()
            filetrainacc.close()
        if phase == 'Test':
            #writer.add_scalar('test_loss', loss, epoch)
            #writer.add_scalar('test_acc', acc, epoch)
            filetestloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetestacc.write(str(epoch) + ' ' + str(acc) + '\n')
            filetestloss.close()
            filetestacc.close()
    else:
        print("\t[%5sing set] Loss: %6f" % (phase, loss))

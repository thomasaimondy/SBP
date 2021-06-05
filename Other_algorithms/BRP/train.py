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

cossim = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)

def train(args, device, train_loader, traintest_loader, test_loader):
    torch.manual_seed(args.seed)

    for trial in range(1,args.trials+1):
        # Network topology
        model = models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, spike_window=args.spike_window,  device=device, thresh=args.thresh, randKill=args.randKill, lens=args.lens, decay=args.decay, dataset=args.dataset)

        if args.cuda:
            model.cuda()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        print(model)
        # Loss function
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,))
        filepath = 'model/' + args.codename.split('-')[0] + '/' + args.codename
        if os.path.exists(filepath+'/model.pth') and args.save_model:
            checkpoint = torch.load(filepath+'/model.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 1
            print('无保存模型，将从头开始训练！')

        for epoch in range(start_epoch, args.epochs + 1):
            # Training
            train_epoch(args, model, device, train_loader, optimizer, loss)

            # Compute accuracy on training and testing set
            print("\nSummary of epoch %d:" % (epoch))
            test_epoch(args, model, device, traintest_loader, loss, 'Train',epoch)
            test_epoch(args, model, device, test_loader, loss, 'Test',epoch)
            if args.save_model:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, filepath+'/model.pth')


def train_epoch(args, model, device, train_loader, optimizer, loss):
    model.train()

    if args.freeze_conv_layers:
        for i in range(model.conv_to_fc):
            for param in model.layers[i].conv.parameters():
                param.requires_grad = False

    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        data, label = data.to(device), label.to(device)#.unsqueeze(1)
        if args.regression:
            targets = label
        elif args.dataset == 'nettalk':
            targets = label.float()
        else:
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)

        optimizer.zero_grad()
        output = model(data, targets)
        # loss_val = loss_function(output, targets)
        loss_val = loss[0](output, loss[1](targets))
        loss_val.backward(retain_graph = True)
        optimizer.step()

def writefile(args, file):
    filepath = 'output/'+args.codename.split('-')[0]+'/'+args.codename
    filetestloss = open(filepath + file, 'a')
    return filetestloss

def test_epoch(args, model, device, test_loader, loss, phase,epoch):
    model.eval()

    test_loss, correct = 0, 0
    # if args.dataset != 'tidigits':
    len_dataset = len(test_loader.dataset)
    total = 0
    # else:
    #     len_dataset = test_loader[1].shape[0]*test_loader[1].shape[1]
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            if args.regression:
                targets = label
            elif args.dataset == 'nettalk':
                targets = label.float()
            else:
                targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1,label.unsqueeze(1).long(), 1.0)

            output = model(data, None)
            if args.dataset == 'nettalk':
                total += 1
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
                    correct += cossim(tem_out, tem_lab)
                else:
                    correct += 0
            else:
                test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                if not args.regression:
                    correct += pred.eq(label.view_as(pred).long()).sum().item()

    loss = test_loss / len_dataset
    if not args.regression:
        if args.dataset == 'nettalk':
            acc = 100. * correct / total
        else:
            acc = 100. * correct / len_dataset
        print("\t[%5sing set] Loss: %6f, Accuracy: %6.2f%%" % (phase, loss, acc))


        filetestloss = writefile(args, '/testloss.txt')
        filetestacc = writefile(args, '/testacc.txt')
        filetrainloss = writefile(args, '/trainloss.txt')
        filetrainacc = writefile(args, '/trainacc.txt')

        if phase == 'Train':
            # writer.add_scalar('train_loss', loss, epoch)
            # writer.add_scalar('train_acc', acc, epoch)
            filetrainloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetrainacc.write(str(epoch) + ' ' + str(acc) + '\n')
        if phase == 'Test':
            # writer.add_scalar('test_loss', loss, epoch)
            # writer.add_scalar('test_acc', acc, epoch)
            filetestloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetestacc.write(str(epoch) + ' ' + str(acc) + '\n')
    else:
        print("\t[%5sing set] Loss: %6f" % (phase, loss))
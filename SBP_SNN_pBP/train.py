import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from SNN import SNN
import time
import os
from tensorboardX import SummaryWriter
from nettalk import Nettalk
from gesture import Gesture
import argparse

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-gpu', type=int, default=3)
parser.add_argument('-seed', type=int, default=3154)
parser.add_argument('-num_epoch', type=int, default=100)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-interval', type=int, default=20, help='interval of loss print during training')
parser.add_argument('-bp_mark', type=int)
parser.add_argument('-hidden_size', type=int, default=500)
parser.add_argument('-alpha', type=float, default=0.1)
parser.add_argument('-task', type=str, default='MNIST', choices=['MNIST', 'NETTalk', 'DVSGesture'])
parser.add_argument('-energy', action='store_true')
parser.add_argument('-sbp', action='store_true')
parser.add_argument('-tensorboard', action='store_true')

opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True

test_scores = []
train_scores = []

if opt.task == 'MNIST':
    if opt.tensorboard:
        writer = SummaryWriter(comment = '-Mni')
    hyperparams = [100, 784, 10, 1e-3, 20, 'MNIST']
    train_dataset = dsets.MNIST(root = './data/mnist', train = True, transform = transforms.ToTensor(), download = True)
    test_dataset = dsets.MNIST(root = './data/mnist', train = False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
elif opt.task == 'NETTalk':
    if opt.tensorboard:
        writer = SummaryWriter(comment = '-Net')
    hyperparams = [5, 189, 26, 1e-3, 20, 'NETTalk']
    train_dataset = Nettalk('train', transform=transforms.ToTensor())
    test_dataset = Nettalk('test', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
elif opt.task == 'DVSGesture':
    if opt.tensorboard:
        writer = SummaryWriter(comment = '-Ges')
    hyperparams = [16, 1024, 11, 1e-4, 20, 'DVSGesture']
    train_dataset = Gesture('train', transform=transforms.ToTensor())
    test_dataset = Gesture('test', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)

print('Dataset: ' + opt.task)
print('Random Seed: {}'.format(opt.seed))
print('Alpha: {}'.format(opt.alpha))
print('Length of Training Dataset: {}'.format(len(train_dataset)))
print('Length of Test Dataset: {}'.format(len(test_dataset)))
print('Build Model')

model = SNN(hyperparams, opt.hidden_size, opt.layers, opt.sbp, opt.bp_mark)
model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams[3])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
sigmoid = torch.nn.Sigmoid()

def train(epoch):
    model.train()
    print('Train Epoch ' + str(epoch + 1))
    start_time = time.time()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if images.size()[0] == hyperparams[0]:
            optimizer.zero_grad()
            images = Variable(images.cuda())
            if opt.task == 'MNIST':
                one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                labels = Variable(one_hot.cuda())
            elif opt.task == 'NETTalk':
                labels = labels.float()
                labels = Variable(labels.cuda())
            elif opt.task == 'DVSGesture':
                labels = labels.float()
                labels = Variable(labels.cuda())

            outputs, e_loss = model(images, labels)
            c_loss = loss_function(outputs, labels)
            loss = c_loss + e_loss * opt.alpha if opt.energy else c_loss
            total_loss += float(loss)
            loss.backward(retain_graph = True)

            optimizer.step()

            if (i + 1) % (len(train_dataset) // (hyperparams[0] * opt.interval)) == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.6f, Time: %.2f' % (epoch + 1, opt.num_epoch, i + 1, len(train_dataset) // hyperparams[0], total_loss / (hyperparams[0] * opt.interval), time.time() - start_time))
                xs = epoch * opt.interval + ((i + 1) // (len(train_dataset) // (hyperparams[0] * opt.interval)))
                if opt.tensorboard:
                    writer.add_scalar('loss_train', total_loss / (hyperparams[0] * opt.interval), xs)
                    writer.add_scalar('time_train', time.time() - start_time, xs)
                start_time = time.time()
                total_loss = 0
    scheduler.step()

def eval(epoch, if_test):
    model.eval()
    correct = 0
    total = 0
    if if_test:
        print('Test Epoch ' + str(epoch + 1))
        loader = test_loader
        test_or_train = 'test'
    else:
        loader = train_loader
        test_or_train = 'train'

    if opt.task == 'MNIST':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            correct += (pred == labels).sum()
        correct = correct.item()
    elif opt.task == 'NETTalk':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images, labels)
            total += 1
            if outputs.max() >= 0.05:
                pos = []
                for label in range(26):
                    if (labels[0, label] != 0) or (outputs[0, label] != 0):
                        pos.append(label)
                tem_out = torch.zeros((1, len(pos)))
                tem_lab = torch.zeros((1, len(pos)))
                for label in range(len(pos)):
                    tem_out[0, label] = outputs[0, pos[label]]
                    tem_lab[0, label] = labels[0, pos[label]]
                correct += cossim(tem_out, tem_lab)
            else:
                correct += 0
    elif opt.task == 'DVSGesture':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images, labels)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            t_label = labels.max(1)[1]
            correct += (pred == t_label).sum()
        correct = correct.item()

    acc = 100.0 * correct / total
    print(test_or_train + ' correct: %d accuracy: %.2f%%' % (correct, acc))
    if opt.tensorboard:
        writer.add_scalar('acc_' + test_or_train, acc, epoch + 1)
    if if_test:
        test_scores.append(acc)
    else:
        train_scores.append(acc)

def main():
    for epoch in range(opt.num_epoch):
        train(epoch)
        if (epoch + 1) % 1 == 0:
            eval(epoch, if_test = True)
        if (epoch + 1) % 20 == 0:
            print('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
            avg = (test_scores[-1] + test_scores[-2] + test_scores[-3] + test_scores[-4] + test_scores[-5] + test_scores[-6] + test_scores[-7] + test_scores[-8] + test_scores[-9] + test_scores[-10]) / 10
            print('Average of Last Ten Test Accuracy : %.2f%%' % (avg))
    if opt.tensorboard:
        writer.close()

if __name__ == '__main__':
    main()
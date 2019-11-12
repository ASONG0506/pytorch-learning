# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time:
# usage: mnist classification
# --------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
sys.path.append("./")

import models.cifar.vgg as vgg
import models.cifar.resnet as resnet
import models.cifar.LeNet as LeNet
import models.cifar.googlenet as GoogLeNet

parser = argparse.ArgumentParser()
parser.add_argument("--lr" , default = 0.1, type=float, help="learning rate")
# parser.add_argument("--resume", action="store_true", help = "resume from scratch")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0
start_epoch = 0

# 1. prepare the data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root = "../../data/", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = "../../data/", train=False, transform=transform_test, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle = False, num_workers = 2)

classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}


# 2. buld the model
# net = vgg.vgg16_bn()
# net = resnet.ResNet50()
# net = LeNet.LeNet()
net = GoogLeNet.GoogLeNet()

net = net.to(device)

# 4. set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD((net.parameters()), lr = args.lr, momentum=0.9, weight_decay = 5e-4)

def train(epoch):
    print("epoch: {}".format(epoch))
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
        print("epoch: {}, ({}/{}), accuracy:{:.3f}%, total_loss:{:.3f}".format(epoch,
                batch_idx, len(trainloader), 100. * correct/total, (train_loss / total)))


def test(epoch):
    global best_acc
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
            print("epoch:{}, batch_idx: {}/{}, accuracy: {:.3f}%, loss:{:.3f}".format(epoch,
                    batch_idx, len(testloader), correct/total * 100. , (test_loss/total)))

    acc = 100. * correct / total
    if acc > best_acc:
        print("saving the model")
        state = {'net':net.state_dict(),
                "acc":acc,
                "epoch":epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/ckpt.pth")
        best_acc = acc

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
import json
import argparse
import os
import time

import torch
import torch.nn as nn
#  import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# AMP
from apex import amp

#  from AlexNet import AlexNet
from model.mobilenetv3 import MobileNetV3_Large


def train():
    #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    #  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #  testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    torch.backends.cudnn.benchmark = True
    #  net = AlexNet()
    net = MobileNetV3_Large().cuda()
    # net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # FOR AMP
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    # END

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(10):
        total_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # FOR AMP
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # END
            # normal loss backward
            #  loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, total_loss / 10))
                total_loss = 0.0

    end_time = time.time()
    print("run time: ", end_time-start_time)
    print('Finished Training...')

    #  torch.save(net.state_dict(), './checkpoints/mobilenetv3-amp.pth')


def main():
    train()


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
#  import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#  from AlexNet import AlexNet
from model.mobilenetv3 import MobileNetV3_Large


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #  net = AlexNet()
    net = MobileNetV3_Large()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(2):
        total_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, total_loss / 20))
                total_loss = 0.0
    print('Finished Training...')

    torch.save(net.state_dict(), './checkpoints/mobilenetv3-temp.pth')


def main():
    train()


if __name__ == '__main__':
    main()

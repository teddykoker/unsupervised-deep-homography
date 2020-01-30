import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Block(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=True):
        super(Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Net(nn.Module):
    def __init__(self, batch_norm=True):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            Block(2, 64, batch_norm),
            Block(64, 128, batch_norm),
            Block(128, 256, batch_norm),
            Block(256, 256, batch_norm),
            Block(256, 256, batch_norm),
            Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(), nn.Linear(4096, 4 * 2),
        )

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)  # combine two images in channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        delta = x.view(-1, 4, 2)
        return delta

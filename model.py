import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


def photometric_loss(delta, img_a, patch_b, corners):
    corners_hat = corners + delta

    # in order to apply transform and center crop,
    # subtract points by top-left corner (corners[N, 0])
    corners = corners - corners[:, 0].view(-1, 1, 2)

    h = kornia.get_perspective_transform(corners, corners_hat)

    h_inv = torch.inverse(h)
    patch_b_hat = kornia.warp_perspective(img_a, h_inv, (128, 128))

    return F.l1_loss(patch_b_hat, patch_b)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Block(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
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
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Net(nn.Module):
    def __init__(self, batch_norm=False):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            Block(2, 64, batch_norm),
            Block(64, 64, batch_norm),
            Block(64, 128, batch_norm),
            Block(128, 128, batch_norm, pool=False),
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4 * 2),
        )

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)  # combine two images in channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        delta = x.view(-1, 4, 2)
        return delta

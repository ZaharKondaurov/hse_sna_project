import torch.nn as nn
import torchvision


def convrelubatch(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


class DyslexiaResNet(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(512, 2)

        # unfreeze backbone layers
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class DyslexiaCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int):
        super().__init__()

        self.num_classes = num_classes

        # number of filters
        cur_img_size = img_size
        f = [3, 8, 16, 16, 32]

        self.crb1 = convrelubatch(f[0], f[1], 7, padding=0)
        cur_img_size -= 6

        self.crb2 = convrelubatch(f[1], f[2], 7, padding=0)
        cur_img_size -= 6

        self.crb3 = convrelubatch(f[2], f[3], 3, padding=1)
        cur_img_size = cur_img_size

        self.crb4 = convrelubatch(f[3], f[4], 3, padding=1)
        cur_img_size = cur_img_size

        self.pool = nn.MaxPool2d(2, 2)
        cur_img_size //= 2

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.drop = nn.Dropout(p=0.25)
        self.fc   = nn.Linear(f[4] * cur_img_size * cur_img_size, 256)
        self.relu = nn.ReLU()

        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.crb1(x)
        x = self.crb2(x)
        x = self.crb3(x)
        x = self.crb4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.head(x)
        return x

import torch
from torch import nn
import torch.nn.functional as F

class ResidualConnection(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()

        self.conv3x3_1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )

        self.batchnorm_1 = nn.BatchNorm2d(num_features=hidden_size)

        self.conv3x3_2 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )

        self.batchnorm_2 = nn.BatchNorm2d(num_features=hidden_size)

    def forward(self, x):
        output = self.conv3x3_1(x)
        output = self.batchnorm_1(output)
        output = self.conv3x3_2(output)
        output = self.batchnorm_2(output)
        output += x  

        return output


class ResidualConnectionWithConv(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()

        self.conv3x3_1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )

        self.batchnorm_1 = nn.BatchNorm2d(num_features=hidden_size)

        self.conv3x3_2 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )

        self.batchnorm_2 = nn.BatchNorm2d(num_features=hidden_size)

        self.conv1x1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hidden_size,
            kernel_size=1,
        )

    def forward(self, x):
        output = self.conv3x3_1(x)
        output = self.batchnorm_1(output)
        output = self.conv3x3_2(output)
        output = self.batchnorm_2(output)
        output += self.conv1x1(x)  

        return output


class ResNet18(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()

        self.conv7x7 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hidden_size,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.batchnorm = nn.BatchNorm2d(num_features=hidden_size)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.firstResidual = nn.Sequential(
            ResidualConnection(in_dim=hidden_size, hidden_size=hidden_size),
            ResidualConnection(in_dim=hidden_size, hidden_size=hidden_size)
        )

        self.bridgeResidual = nn.Sequential(
            ResidualConnectionWithConv(in_dim=hidden_size, hidden_size=hidden_size),
            ResidualConnection(in_dim=hidden_size, hidden_size=hidden_size)
        )

        self.secondResidual = nn.Sequential(
            self.bridgeResidual,
            self.bridgeResidual,
            self.bridgeResidual
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.FC = nn.Linear(hidden_size, 10)  
        self.loss = nn.CrossEntropyLoss() 

    def forward(self, x, y):
        output = self.conv7x7(x) 
        output = self.batchnorm(output)
        output = self.maxpool(output)
        output = self.firstResidual(output)
        output = self.secondResidual(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)  

        y_hat = self.FC(output)
        loss = self.loss(y_hat, y)

        return y_hat, loss

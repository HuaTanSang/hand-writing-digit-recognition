import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, ch1x1red):
        super(InceptionModule, self).__init__()

        # Nhánh 1
        self.branch1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=ch1x1, 
            kernel_size=1
        )

        # Nhánh 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=ch3x3red, kernel_size=1),
            nn.Conv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1)
        )

        # Nhánh 3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            nn.Conv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, padding=2)
        )

        # Nhánh 4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=ch1x1red, kernel_size=1)
        )

    def forward(self, x):

        # branch1 = F.relu(self.branch1(x))
        # branch2 = F.relu(self.branch2(x)) 
        # branch3 = F.relu(self.branch3(x))
        # branch4 = F.relu(self.branch4(x)) 

        branch1 = self.branch1(x)
        branch2 = self.branch2(x) 
        branch3 = self.branch3(x)
        branch4 = self.branch4(x) 

        outputs = [branch1, branch2, branch3, branch4]

        return torch.cat(outputs, dim=1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()

        # Bước 1 (Các "bước" được định nghĩa theo trang d2l)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bước 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1) 
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
        # Bước 3
        self.inception1 = nn.Sequential(
            InceptionModule(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, ch1x1red=32),
            InceptionModule(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, ch1x1red=64)
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bước 4
        self.inception4 = nn.Sequential(
            InceptionModule(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, ch1x1red=64),
            InceptionModule(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, ch1x1red=64),
            InceptionModule(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, ch1x1red=64),
            InceptionModule(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, ch1x1red=64),
            InceptionModule(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, ch1x1red=128)
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Bước 5
        self.inception5 = nn.Sequential(
            InceptionModule(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, ch1x1red=128),
            InceptionModule(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, ch1x1red=128)
        )

        # Bước 6: Average Pooling, Dropout và Fully Connected
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(1024, 10)  # Điều chỉnh output cho phù hợp

        self.loss = nn.CrossEntropyLoss() 

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Bước 1
        x = self.conv1(x)
        x = self.maxpool1(x)

        # Bước 2 
        x = self.conv2_1(x)
        x = self.conv2_2(x) 
        x = self.maxpool2(x)

        # Bước 3 
        x = self.inception1(x)
        x = self.maxpool3(x)

        # Bước 4 
        x = self.inception4(x)
        x = self.maxpool4(x)

        # Bước 5
        x = self.inception5(x) 

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten trước khi đưa vào fully connected
        x = self.dropout(x)
        
        y_hat = self.fc(x)
        loss = self.loss(y_hat, y)
        return y_hat, loss

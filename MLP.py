import torch 
from torch import nn 

class One_Layer_MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.MLP = nn.Linear(784, 10) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images: torch.Tensor):
        images = images.reshape((images.shape[0], -1))  
        features = self.MLP(images)
        results = self.softmax(features)
        return results

class Three_Layer_MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.MLP1 = nn.Linear(784, 128)
        self.MLP2 = nn.Linear(128, 64)
        self.MLP3 = nn.Linear(64, 10)  
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, images: torch.Tensor):
        images = images.reshape((images.shape[0], -1)) 
        features = self.MLP1(images)
        # features = self.dropout(features)
        features = self.ReLU(features)
        features = self.MLP2(features)
        # features = self.dropout(features)
        features = self.ReLU(features)
        features = self.MLP3(features)  
        results = self.softmax(features)

        return results

import numpy as np
import torch 
import torch.nn as nn 

from tqdm import tqdm 
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from dataset import MNISTDataset
from utils import preprocessing_label, collate_fn
from MLP import One_Layer_MLP, Three_Layer_MLP
from GoogLeNet import GoogleNet
from ResNet18 import ResNet18
from LeNet import LeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train = MNISTDataset(
    image_path=r'D:\UIT\Tài liệu\Năm 3\Deep Learning\mnist\train-images.idx3-ubyte',
    label_path=r'D:\UIT\Tài liệu\Năm 3\Deep Learning\mnist\train-labels.idx1-ubyte'
    )

    test = MNISTDataset(
        image_path=r'D:\UIT\Tài liệu\Năm 3\Deep Learning\mnist\t10k-images.idx3-ubyte',
        label_path=r'D:\UIT\Tài liệu\Năm 3\Deep Learning\mnist\t10k-labels.idx1-ubyte'
    )

    train_loader = DataLoader(
        train, batch_size=64,
        shuffle=True, 
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test, batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
    )




































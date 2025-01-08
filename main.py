import numpy as np
import torch 
import torch.nn as nn 
import os 

import torch.optim.optimizer
from tqdm import tqdm 
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from shutil import copyfile

from dataset import MNISTDataset
from utils import preprocessing_label, collate_fn
from MLP import One_Layer_MLP, Three_Layer_MLP
from GoogLeNet import GoogleNet
from ResNet18 import ResNet18
from LeNet import LeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_model(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer):
    model.train() 

    running_loss = .0 
    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(dataloader)) as pb: 
        for it, items in enumerate(dataloader): 
            input_ids = items['input_ids'].to(device)
            labels = items['labels'].to(device) 

            _, loss = model(input_ids, labels)

            # Back propagation 
            optim.zero_grad() 
            loss.backward() 
            optim.step() 
            running_loss += loss.item() 

            # Update training status
            pb.set_postfix(loss=running_loss / (it + 1))
            pb.update() 

def compute_scores(predicted_label: list, true_label: list) -> list:
    metrics = {
        "f1" : f1_score, 
        "accuracy" : accuracy_score
    }

    scores = {} 

    for metric_name in metrics:
        scorer = metrics[metric_name]
        scores[metric_name] = scorer(true_label, predicted_label, averange='micro')
    
    return scores 


def evaluate_model(epoch: int, model: nn.Module, dataloader: DataLoader) -> dict: 
    model.eval() 
    all_true_labels = [] 
    all_predicted_labels = [] 
    scores = {} 

    with tqdm(desc='Epoch %d - Evaluating' & epoch, unit='it', total=len(dataloader)) as pb: 
        for items in dataloader: 
            input_ids = items['input_ids'].to(device)
            labels = items['lables'].to(device)

            with torch.no_grad(): 
                logits, _ = model(input_ids, labels)
            
            prediction = logits.argmax(dim=-1).long() 

            labels = labels.view(-1).cpu().numpy() 

            all_predicted_labels.extend(prediction) 
            all_true_labels.extend(labels) 

            pb.update() 

    scores = compute_scores(all_predicted_labels, all_true_labels)

def save_checkpoint(dict_to_save: dict, checkpoint_dir: str): 
    if not os.path.isdir(checkpoint_dir): 
        os.mkdir(checkpoint_dir)
    torch.save(dict_to_save, os.path.join(f"{checkpoint_dir}", "last_model.pth"))

def main(in_dim: int, hidden_size: int, learning_rate: float, checkpoint_dir: str):

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

    epoch = 0
    allowed_patience = 5
    best_score = 0 
    compared_score = "f1"
    patience = 0 

    model = LeNet() 

    model = model.to(device) 
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while True: 
        train(epoch, model, train_loader, optim)
        # validate
        scores = evaluate_model(epoch, model, test_loader) 
        print(f"Scores: ", scores)
        score = scores[compared_score]

        # Prepare for next epoch
        is_best_model = False 
        if score > best_score: 
            best_score = score
            patience = 0 
            is_best_model = True 
        else: 
            patience += 1 
        
        if patience == allowed_patience: 
            exit_train = False 
        

        save_checkpoint({
            "epoch": epoch, 
            "best_score": best_score, 
            "patience": patience,
            "state_dict": model.state_dict(), 
            "optimizer": optim.state_dict() 
        }, checkpoint_dir)


        if is_best_model: 
            copyfile(
                os.path.join(checkpoint_dir, "last_model.pth"), 
                os.path.join(checkpoint_dir, "best_model.pth")
            )
        
        if exit_train: 
            break 

        epoch += 1 
    
if __name__ == "__main__": 
    main(
        in_dim=1, hidden_size=64, learning_rate=0.001, 
        checkpoint_dir=r'D:\UIT\Tài liệu\Năm 3\Deep Learning\hand-writing-digit-recognition\checkpoint'
    )



































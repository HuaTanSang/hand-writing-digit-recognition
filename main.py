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
from directory import train_image_path, test_image_path, train_label_path, test_label_path, checkpoint_path, MLP_1_Layer_checkpoint, MLP_3_Layer_checkpoint, LeNet_checkpoint, GoogLeNet_checkpoint, ResNet_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_model(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer):
    model.train() 

    running_loss = .0 
    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(dataloader)) as pb: 
        for it, (image, label) in enumerate(dataloader): 
            image = image.to(device)
            label = label.to(device) 

            _, loss = model(image, label)

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
        if scorer == accuracy_score: 
            scores[metric_name] = scorer(true_label, predicted_label)
        else: 
            scores[metric_name] = scorer(true_label, predicted_label, average='weighted') 
    
    return scores 


def evaluate_model(epoch: int, model: nn.Module, dataloader: DataLoader) -> dict: 
    model.eval() 
    all_true_labels = [] 
    all_predicted_labels = [] 
    scores = {} 

    with tqdm(desc='Epoch %d - Evaluating' % epoch, unit='it', total=len(dataloader)) as pb: 
        for image, label in dataloader: 
            image = image.to(device)
            label = label.to(device)
                        
            with torch.no_grad(): 
                logits, _ = model(image, label)
            
        
            prediction = logits.argmax(dim=-1).long().cpu().numpy()

            label = label.view(-1).cpu().numpy() 

            all_predicted_labels.extend(prediction) 
            all_true_labels.extend(label) 
        
            pb.update() 

    scores = compute_scores(all_predicted_labels, all_true_labels)
    return scores 

def save_checkpoint(dict_to_save: dict, checkpoint_dir: str): 
    if not os.path.isdir(checkpoint_dir): 
        os.mkdir(checkpoint_dir)
    torch.save(dict_to_save, os.path.join(f"{checkpoint_dir}", "last_model.pth"))

def main(in_dim: int, hidden_size: int, learning_rate: float):
    
    train_dataset = MNISTDataset(
    image_path=train_image_path,
    label_path=train_label_path
    )

    test_dataset = MNISTDataset(
        image_path=test_image_path,
        label_path=test_label_path
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64,
        shuffle=True, 
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    epoch = 0
    allowed_patience = 4
    best_score = 0 
    compared_score = "f1"
    patience = 0 
    exit_train = False

    model_list = {
        One_Layer_MLP: MLP_1_Layer_checkpoint, 
        Three_Layer_MLP: MLP_3_Layer_checkpoint, 
        LeNet: LeNet_checkpoint, 
        GoogleNet: GoogLeNet_checkpoint, 
        ResNet18: ResNet_checkpoint
    }

    for model in model_list: 
        checkpoint_dir = model_list[model]

        model = ResNet18(in_dim=1, hidden_size=64)

        model = model.to(device) 
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        while True: 
            train_model(epoch, model, train_loader, optim)
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
                exit_train = True
            

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
        in_dim=1, hidden_size=64, learning_rate=0.001
    )



































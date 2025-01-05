import torch
from typing import List 


def collate_fn(items: List[dict]) -> torch.Tensor:
    images = [torch.Tensor(item['image']).unsqueeze(0) for item in items]  
    labels = [item['label'] for item in items]  
    
    images = torch.stack(images)  
    labels = torch.tensor(labels, dtype=torch.long) 

    return images, labels


def preprocessing_label(label: int) -> torch.Tensor:
    new_label = torch.zeros(10)
    new_label[label] = 1

    return new_label
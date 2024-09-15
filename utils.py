import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
def load_model_checkpoint(path:str, model:nn.Module, optimizer:torch.optim.Optimizer ):
    if os.path.isfile(path):
        print("file found Loading checkpoint")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f" checkpoint loaded from {path}")
    else:
        print("checkpoint not found ")
        return 0
    
def save_model_checkpoint(path:str, model:nn.Module, optimizer:torch.optim.Optimizer ):
    checkpoint = {
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"checkpoint saved at path {path} ")


def generate_dataset(base_data_dir_path, IMG_SIZE=224):
    manual_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor()
    ])

    ## create dataset
    train_dataset=datasets.ImageFolder(root=f'./{base_data_dir_path}/train', 
                                    transform=manual_transform)

    valid_dataset=datasets.ImageFolder(root=f'./{base_data_dir_path}/val', 
                                    transform=manual_transform)

    test_dataset=datasets.ImageFolder(root=f'./{base_data_dir_path}/test', 
                                    transform=manual_transform)
    
    return train_dataset, valid_dataset, test_dataset


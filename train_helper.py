import torch.nn as nn
import torch
from typing import Tuple
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm

loss_fn=torch.nn.CrossEntropyLoss()
def train_loop(rnk:int, args : dict, model : nn.Module, 
               device : str, optimizer : torch.optim.Optimizer,
               train_dataloader : DataLoader,
               valid_dataloader:DataLoader):
    torch.manual_seed(args.seed+rnk)

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss, train_acc=train_step(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=device,
                                      args=args,
                                      epoch=epoch)
        test_loss, test_acc=valid_step(model=model,
                                  dataloader=valid_dataloader,
                                  loss_fn=loss_fn,
                                  device=device,
                                  args=args,
                                  epoch=epoch)

        print(f"PID {os.getpid()} | Epoch: {epoch+1} | train_loss: {train_loss} | train_acc: {train_acc} | test_loss: {test_loss} | test_acc: {test_acc}")




def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               args:dict, 
               epoch:int) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0.0,  0.0

    for batch_id, (X,y) in enumerate(dataloader):
        X,y  = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # loss
        loss = loss_fn(y_pred, y)
        train_loss+=loss.item()
        # zero_out grad
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # optimizer step
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

        if batch_id % args.log_interval == 0:
            print(f"PID {os.getpid()}  Train Loss {train_loss} \n Train acc {train_acc}")
            print("+"*50)
    
        
            
    
    
    train_loss/=len(dataloader)
    train_acc/=len(dataloader)
    print(f"epoch {epoch} PID {os.getpid()}  Train Loss {train_loss} \n Train acc {train_acc}")
    print("+"*50)
    return train_loss, train_acc







def valid_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device, 
               args:dict, 
               epoch:int) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0.0,  0.0

    for batch_id, (X,y) in enumerate(dataloader):
        X,y  = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # loss
        loss = loss_fn(y_pred, y)
        test_loss+=loss.item()
        

        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
        test_acc += (y_pred_class==y).sum().item()/len(y_pred)
        if batch_id % args.log_interval == 0:
            print(f"PID {os.getpid()}  Validation Loss {test_loss} \n Validation acc {test_acc}")
            print("+"*50)
    test_loss/=len(dataloader)
    test_acc/=len(dataloader)
    print(f"epoch {epoch} PID {os.getpid()}  Validation Loss {test_loss} \n Validation acc {test_acc}")
    print("+"*50)
    return test_loss, test_acc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import torch.multiprocessing as mp
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
from train_helper import train_loop
from model_builder import TinyVGG
from utils import load_model_checkpoint, save_model_checkpoint, generate_dataset

def main():
    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 1)')
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    parser.add_argument(
        "--resume",default=True, help="Flag if you want to resume training or not!!!"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # create model and setup mp

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mp.set_start_method('spawn')
    model = TinyVGG(input_channel=3, hidden_units=10, output_shape=5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    summary(model, input_size=(3, 224,224))
    if args.resume:
        if not os.path.isfile('./model/model_checkoint_path.pth'):
            print("No checkpoint found to resume from")
        else:
            print("resuming from checkpoint ")
            load_model_checkpoint(path = './model/model_checkpoint_path.pth',
                                model = model, 
                                optimizer = optimizer)
    

    model.share_memory()

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }

    # create dataset and dataloader
    
    train_dataset, valid_dataset, test_dataset=generate_dataset( base_data_dir_path='data',
                                                                IMG_SIZE=224)
                 
    ## create data loader
    train_data_loader=DataLoader(dataset=train_dataset,
                                **kwargs)

    val_data_loader=DataLoader(dataset=valid_dataset,
                                batch_size=kwargs['batch_size'],
                                pin_memory=kwargs['pin_memory'],
                                num_workers=kwargs["num_workers"],
                                shuffle=False)

    test_data_loader=DataLoader(dataset=test_dataset,
                                batch_size=kwargs['batch_size'],
                                pin_memory=kwargs['pin_memory'],
                                num_workers=kwargs["num_workers"],
                                shuffle=False)

    processes = []

    # https://github.com/pytorch/examples/blob/main/mnist_hogwild/main.py
    for rank in range(args.num_processes):
        p = mp.Process(target=train_loop,args=(rank,
                                                args,
                                                model,
                                                device, 
                                                optimizer, 
                                                train_data_loader,
                                                 val_data_loader ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    save_model_checkpoint(path='./model/model_checkpoint_path.pth',model=model, optimizer=optimizer)



# save model ckpt

 
if __name__ == "__main__":
    main()
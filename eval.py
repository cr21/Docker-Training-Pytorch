import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model_builder import TinyVGG
from utils import generate_dataset, load_model_checkpoint
import os
from train_helper import valid_step



def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )

    parser.add_argument(
        "--test-batch-size", type=int, default=16, metavar="S", help="batch size"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }
    
    _,_, test_dataset = generate_dataset(base_data_dir_path='data',
                                         IMG_SIZE=224)
    test_dataloader = DataLoader(dataset=test_dataset,**kwargs)
    class_names = test_dataset.classes
    print(f"class_names {class_names}")
    # create model and load state dict
    model =  TinyVGG(input_channel=3, hidden_units=10, output_shape=5).to(device)
    if not os.path.isfile(f'model/model_checkpoint_path.pth'):
        print("Model does not exists at location")
    else:
        print("Checkpoint found loading from checkpoint")
        checkpt = torch.load('model/model_checkpoint_path.pth')
        model.load_state_dict(checkpt['model_state_dict'])
    
    # test epoch function call
    test_loss, test_acc = valid_step(model=model,
               dataloader = test_dataloader,
               loss_fn = torch.nn.CrossEntropyLoss(),
               device = device, 
               args = args, 
               epoch = 0)
    eval_results = {'test_loss':test_loss, 'test_acc':test_acc}
    with (Path(args.save_dir) / "model" / "eval_results.json").open("w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()

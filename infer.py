import json
import time
import random
import torch
from torchvision.io import read_image
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from model import Net
from model_builder import TinyVGG
import os
from utils import generate_dataset
from typing import List, Tuple
import matplotlib.pyplot as plt

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    results_dir:str,
    class_names: List[str] = None,
    image_size: Tuple[int, int] = (224, 224),
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Makes a prediction on a target image with a trained model and plots the image.

    
        # Get a random list of image paths from test set
        import random
        num_images_to_plot = 3
        test_image_path_list = list(Path(test_path).glob("*/*.jpg")) # get list all image paths from test data 
        test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                               k=num_images_to_plot) # randomly select 'k' image paths to pred and plot
        
        # Make predictions on and plot the images
        for image_path in test_image_path_sample:
            pred_and_plot_image(model=model, 
                                image_path=image_path,
                                class_names=class_names,
                                # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                                image_size=(224, 224))
            plt.show()
    """
    # 1. Load in image and convert the tensor values to float32
    target_image = read_image(str(image_path)).type(torch.float32)
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

     # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    result_path = f"{results_dir}/{image_path.stem}_{class_names[target_image_pred_label.cpu()]}.png"
    plt.savefig(result_path)


def infer(model, save_dir , class_names, num_samples=5 ):
    model.eval()
    results_dir = Path(save_dir) / "results"
    test_image_path_list = list(Path('data/test/').glob("*/*.jpg"))
    results_dir.mkdir(parents=True, exist_ok=True)
    test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                               k=num_samples) # randomly select 'k' image paths to pred and plot
    
    manual_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model, 
                            image_path=image_path,
                            results_dir=results_dir,
                            class_names=class_names,
                            transform=manual_transform, # optionally pass in a specified transform from our pretrained model weights
                            image_size=(224, 224))
        plt.show()


def main():
    save_dir = "responses"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # init model and load checkpoint here
    
    model = TinyVGG(input_channel=3, hidden_units=10, output_shape=5).to(device)

    if not os.path.isfile(f'model/model_checkpoint_path.pth'):
        print("Model does not exists at location")
    else:
        print("Checkpoint found loading from checkpoint")
        checkpt = torch.load('model/model_checkpoint_path.pth')
        model.load_state_dict(checkpt['model_state_dict'])
        print("Model loaded from checkpoints")
    class_names=['bottomwear', 'eyewear', 'footwear', 'handbag', 'topwear']
    infer(model, save_dir, class_names)
    print("Inference completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    main()

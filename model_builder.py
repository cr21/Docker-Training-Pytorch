import torch
import torch.nn as nn
import torch.nn.functional as F
class TinyVGG(nn.Module):
    """
    Creates the TinyVGG architecture
    """
    def __init__(self, input_channel: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*53*53, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.classifier(x)
        return x



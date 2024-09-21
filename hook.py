import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define a simple CNN for image classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hook for recording activations
class ActivationHook:
    def __init__(self, name):
        self.name = name
        self.activations = None

    def __call__(self, module, input, output):
        self.activations = output.detach()

# Hook for recording gradients
class GradientHook:
    def __init__(self, name):
        self.name = name
        self.gradients = None

    def __call__(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

# Function to add hooks to a model
def add_hooks(model):
    activation_hooks = {}
    gradient_hooks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            activation_hooks[name] = ActivationHook(name)
            module.register_forward_hook(activation_hooks[name])

            gradient_hooks[name] = GradientHook(name)
            module.register_backward_hook(gradient_hooks[name])

    return activation_hooks, gradient_hooks

# Training function
def train(model, train_loader, criterion, optimizer, device, hooks):
    model.train()
    activation_hooks, gradient_hooks = hooks

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')

            # Print activation and gradient statistics
            for name, hook in activation_hooks.items():
                if hook.activations is not None:
                    print(f'{name} activation: mean={hook.activations.mean():.4f}, std={hook.activations.std():.4f}')

            for name, hook in gradient_hooks.items():
                if hook.gradients is not None:
                    print(f'{name} gradient: mean={hook.gradients.mean():.4f}, std={hook.gradients.std():.4f}')

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Add hooks to the model
    hooks = add_hooks(model)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, trainloader, criterion, optimizer, device, hooks)

    print("Training finished")

if __name__ == "__main__":
    main()


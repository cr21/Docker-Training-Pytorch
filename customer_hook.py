import os
import json
import time
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

class CustomLogger:
    def __init__(self, project_name, run_name, log_dir="./logs"):
        self.project_name = project_name
        self.run_name = run_name
        self.log_dir = os.path.join(log_dir, project_name, run_name)
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = open(os.path.join(self.log_dir, "metrics.jsonl"), "a")
        
    def log(self, metrics, step=None):
        if step is None:
            step = len(self.metrics["step"]) if "step" in self.metrics else 0
        
        metrics["step"] = step
        metrics["time"] = time.time() - self.start_time
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        self.log_file.write(json.dumps(metrics) + "\n")
        self.log_file.flush()
    
    def log_hyperparameters(self, hyperparameters):
        with open(os.path.join(self.log_dir, "hyperparameters.json"), "w") as f:
            json.dump(hyperparameters, f, indent=2)
    
    def save_model(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.log_dir, f"{name}.pth"))
    
    def plot_metric(self, metric_name):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["step"], self.metrics[metric_name])
        plt.title(f"{metric_name} over time")
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.savefig(os.path.join(self.log_dir, f"{metric_name}.png"))
        plt.close()
    
    def plot_all_metrics(self):
        for metric_name in self.metrics.keys():
            if metric_name not in ["step", "time"]:
                self.plot_metric(metric_name)
    
    def finish(self):
        self.log_file.close()
        self.plot_all_metrics()
        
        # Save a summary of the run
        summary = {
            "project_name": self.project_name,
            "run_name": self.run_name,
            "duration": time.time() - self.start_time,
            "metrics": {k: v[-1] for k, v in self.metrics.items() if k not in ["step", "time"]}
        }
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

# Example usage
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# Training function
def train(model, train_loader, optimizer, criterion, logger, epochs):
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                accuracy = (output.argmax(dim=1) == target).float().mean()
                logger.log({
                    "loss": loss.item(),
                    "accuracy": accuracy.item()
                })
        
        logger.log({"epoch": epoch})
    
    logger.save_model(model, "final_model")

# Main function
def main():
    # Set up data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Set up model and optimizer
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Set up logger
    logger = CustomLogger("MNIST_Classification", "run_1")
    
    # Log hyperparameters
    hyperparameters = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": 64,
        "epochs": 5,
        "model_type": "SimpleLinear"
    }
    logger.log_hyperparameters(hyperparameters)

    # Train the model
    train(model, train_loader, optimizer, criterion, logger, epochs=5)

    # Finish logging
    logger.finish()

if __name__ == "__main__":
    main()

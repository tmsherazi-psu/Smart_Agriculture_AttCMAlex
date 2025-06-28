import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn


# Define the AttCM-Alex model
class AttCM_AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AttCM_AlexNet, self).__init__()

        # AlexNet backbone
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

        # Attention Module (self-attention and channel attention)
        self.attention_module = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Pass through AlexNet
        x = self.alexnet.features(x)

        # Apply Attention module
        attention = self.attention_module(x)
        x = x * attention  # Apply attention to the features

        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier(x)
        return x


# Define a custom agent class for training and evaluation
class Agent_AttCM_Alex:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(config['device'])
        self.config = config
        self.epoch = 0

    def train(self, data_loader, loss_function):
        self.model.train()
        for epoch in range(self.config['n_epoch']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = loss_function(outputs, labels)

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()

                running_loss += loss.item()

                if i % self.config['save_interval'] == 0:  # Save model at intervals
                    print(f"Epoch [{epoch + 1}/{self.config['n_epoch']}], Step [{i + 1}], Loss: {loss.item():.4f}")

            # Step the scheduler
            self.scheduler.step()
            print(f'Epoch [{epoch + 1}/{self.config['
            n_epoch
            ']}], Average Loss: {running_loss / len(data_loader):.4f}')

            def evaluate(self, data_loader, loss_function):
                self.model.eval()
                total_dice_score = 0.0
                with torch.no_grad():
                    for inputs, labels in data_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        # Forward pass
                        outputs = self.model(inputs)

                        # Compute Dice score or any other evaluation metrics
                        dice_score = self.compute_dice_score(outputs, labels)
                        total_dice_score += dice_score

                average_dice_score = total_dice_score / len(data_loader)
                return average_dice_score

            def compute_dice_score(self, outputs, labels):
                smooth = 1e-5
                outputs = torch.sigmoid(outputs)
                intersection = (outputs * labels).sum()
                dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
                return dice.item()

        # Configuration for AttCM-AlexNet
        config = {
            'device': 'cuda:0',
            'lr': 1e-3,
            'lr_gamma': 0.9999,
            'n_epoch': 50,  # Updated to 50 iterations based on the provided details
            'batch_size': 32,
            'save_interval': 10,
            'evaluate_interval': 10,
            'optimizer': 'AdamW',  # Updated to AdamW optimizer
            'scheduler': 'CyclicLR',  # Updated to CyclicLR scheduler
            'loss_function': 'Smoothed Cross-Entropy Loss',  # Updated loss function
        }

        # Initialize the dataset (adjust with actual dataset)
        dataset = YourCustomDataset("image_path", "label_path")  # Replace with your dataset path
        device = torch.device(config['device'])
        model = AttCM_AlexNet(num_classes=2).to(device)  # Assuming binary classification for simplicity

        # Optimizer and Scheduler
        optimizer = AdamW(model.parameters(), lr=config['lr'])
        scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=2000, mode='triangular2')

        # Loss function (e.g., CrossEntropyLoss for classification)
        loss_function = torch.nn.CrossEntropyLoss()

        # Initialize the agent for AttCM-Alex
        agent = Agent_AttCM_Alex(model, optimizer, scheduler, config)

        # Set up data loader (adjust with actual DataLoader and dataset)
        data_loader = DataLoader(dataset, shuffle=True, batch_size=config['batch_size'])

        # Train the model
        agent.train(data_loader, loss_function)

# Import required libraries
import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Apply CLAHE for contrast enhancement
def apply_clahe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

# Dataset loader
def Dataset_loader(DIR, RESIZE, use_clahe=True):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
            if use_clahe:
                img = apply_clahe(img)
            IMG.append(np.array(img))
    return IMG

# Load dataset
cucumber_train = np.array(Dataset_loader('/content/drive/MyDrive/data/Train/cucumber', 256))
banana_train = np.array(Dataset_loader('/content/drive/MyDrive/data/Train/banana', 256))
tomato_train = np.array(Dataset_loader('/content/drive/MyDrive/data/Train/tomato', 256))

cucumber_test = np.array(Dataset_loader('/content/drive/MyDrive/data/Validation/cucumber', 256))
banana_test = np.array(Dataset_loader('/content/drive/MyDrive/data/Validation/banana', 256))
tomato_test = np.array(Dataset_loader('/content/drive/MyDrive/data/Validation/tomato', 256))

# Create labels
cucumber_train_label = np.zeros(len(cucumber_train))
banana_train_label = np.ones(len(banana_train))
tomato_train_label = np.full(len(tomato_train), 2)

cucumber_test_label = np.zeros(len(cucumber_test))
banana_test_label = np.ones(len(banana_test))
tomato_test_label = np.full(len(tomato_test), 2)

# Merge and shuffle
X_train = np.concatenate((cucumber_train, banana_train, tomato_train), axis=0)
Y_train = np.concatenate((cucumber_train_label, banana_train_label, tomato_train_label), axis=0)
X_test = np.concatenate((cucumber_test, banana_test, tomato_test), axis=0)
Y_test = np.concatenate((cucumber_test_label, banana_test_label, tomato_test_label), axis=0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train, Y_train = X_train[s], Y_train[s]
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test, Y_test = X_test[s], Y_test[s]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# Wrap into DataLoader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Label Smoothing CrossEntropy Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=3):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets = targets.view(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        nll_loss = F.nll_loss(log_preds, targets, reduction='none')
        smooth_loss = -log_preds.mean(dim=-1)
        loss = (1 - self.alpha) * nll_loss + self.alpha * smooth_loss
        return loss.mean()

# Attention Convolution Module (AttCM)
class AttCM(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(AttCM, self).__init__()

        # Feature transformation
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=1)

        # Convolution Branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Attention Branch
        self.q_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        conv_out = self.conv_branch(x)

        B, C, H, W = x.size()
        q = self.q_conv(x).view(B, C, -1)
        k = self.k_conv(x).view(B, C, -1).transpose(-2, -1)
        v = self.v_conv(x).view(B, C, -1)

        attn_weights = self.softmax(torch.bmm(k, q))  # Scaled dot-product attention
        attn_out = torch.bmm(v, attn_weights).view(B, C, H, W)

        combined = self.alpha * conv_out + self.beta * attn_out
        return combined

# Modified AlexNet with AttCM
class AttCMAlex(nn.Module):
    def __init__(self, num_classes=3):
        super(AttCMAlex, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        features = list(alexnet.features.children())

        self.layer1 = nn.Sequential(*features[0:2])  # Conv1 + ReLU
        self.pool1 = features[2]  # MaxPool

        self.layer2 = nn.Sequential(*features[3:5])  # Conv2 + ReLU
        self.pool2 = features[5]  # MaxPool

        self.attcm = AttCM(in_channels=192, out_channels=384)
        self.post_attcm = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.attcm(x)
        x = self.post_attcm(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10):
    device = next(model.parameters()).device
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if scheduler:
            scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function
def evaluate_model(model, test_loader):
    y_true, y_pred = [], []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cucumber', 'Banana', 'Tomato'],
                yticklabels=['Cucumber', 'Banana', 'Tomato'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plotting function
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()

# Configuration dictionary
config = {
    'model_path': 'attcm_alexnet_final.pth',
    'device': "cuda:0",
    'lr': 1e-3,
    'n_epoch': 50,
    'batch_size': 32,
    'gamma': 0.9999,
}

# Main function
def main():
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    model = AttCMAlex(num_classes=3).to(device)
    criterion = LabelSmoothingCrossEntropy(alpha=0.1, num_classes=3)
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = ExponentialLR(optimizer, gamma=config['gamma'])

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler=scheduler, epochs=config['n_epoch']
    )

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    evaluate_model(model, test_loader)

    torch.save(model.state_dict(), config['model_path'])

if __name__ == "__main__":
    main()
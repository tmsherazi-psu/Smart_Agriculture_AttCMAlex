# Import Required Libraries
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# --- Custom Label Smoothing Loss ---
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=3):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Convert one-hot targets to class indices
        targets = targets.argmax(dim=1)
        # Smooth labels
        targets = torch.clamp(targets.long(), 0, self.num_classes - 1)
        n = inputs.size()[-1]
        log_preds = F.log_softmax(inputs, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, targets, reduction='mean')
        return (1 - self.alpha) * nll + self.alpha * (loss / n)


# --- Attention Convolution Module (AttCM) ---
class AttCM(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super(AttCM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)

        # Convolution Branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        # Attention Branch
        self.q_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.k_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.v_conv = nn.Conv2d(256, 256, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        # Learnable parameters for combining branches
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Initial feature transformation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Convolution Branch
        conv_out = self.conv_branch(x)

        # Attention Branch
        B, C, H, W = x.size()
        q = self.q_conv(x).view(B, C, -1)
        k = self.k_conv(x).view(B, C, -1).transpose(-2, -1)
        v = self.v_conv(x).view(B, C, -1)

        attn_weights = self.softmax(torch.bmm(k, q))  # Scaled dot-product attention
        attn_out = torch.bmm(v, attn_weights).view(B, C, H, W)

        # Combine both branches
        combined = self.alpha * conv_out + self.beta * attn_out
        return combined


# --- Modified AlexNet with AttCM ---
class AttCMAlex(nn.Module):
    def __init__(self, num_classes=3):
        super(AttCMAlex, self).__init__()

        # Load pre-trained AlexNet
        alexnet = models.alexnet(pretrained=True)
        features = list(alexnet.features.children())

        # Replace three 3x3 conv layers with AttCM
        self.layer1 = nn.Sequential(*features[0:2])  # Conv1 + ReLU
        self.pool1 = features[2]  # MaxPool2d

        self.layer2 = nn.Sequential(*features[3:5])  # Conv2 + ReLU
        self.pool2 = features[5]  # MaxPool2d

        self.attcm = AttCM(in_channels=192, out_channels=384)  # Insert AttCM
        self.post_attcm = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully connected layers (expanded from original AlexNet)
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


# --- Dataset Loading & Transformation ---
def load_data():
    base_dir = "/content/drive/MyDrive/data"

    def load_images_from_folder(folder):
        images = []
        for filename in tqdm(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            if img_path.endswith(".png"):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
        return np.array(images)

    train_cucumber_dir = os.path.join(base_dir, "Train/cucumber")
    train_banana_dir = os.path.join(base_dir, "Train/banana")
    train_tomato_dir = os.path.join(base_dir, "Train/tomato")

    test_cucumber_dir = os.path.join(base_dir, "Validation/cucumber")
    test_banana_dir = os.path.join(base_dir, "Validation/banana")
    test_tomato_dir = os.path.join(base_dir, "Validation/tomato")

    cucumber_train = load_images_from_folder(train_cucumber_dir)
    banana_train = load_images_from_folder(train_banana_dir)
    tomato_train = load_images_from_folder(train_tomato_dir)

    cucumber_test = load_images_from_folder(test_cucumber_dir)
    banana_test = load_images_from_folder(test_banana_dir)
    tomato_test = load_images_from_folder(test_tomato_dir)

    # Labels: 0 - Healthy (cucumber), 1 - Diseased (banana), 2 - Diseased (tomato)
    cucumber_train_label = np.zeros(len(cucumber_train))
    banana_train_label = np.ones(len(banana_train))
    tomato_train_label = 2 * np.ones(len(tomato_train))

    cucumber_test_label = np.zeros(len(cucumber_test))
    banana_test_label = np.ones(len(banana_test))
    tomato_test_label = 2 * np.ones(len(tomato_test))

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

    Y_train = np.eye(3)[Y_train.astype(int)]
    Y_test = np.eye(3)[Y_test.astype(int)]

    X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies


# --- Evaluation Function ---
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.argmax(dim=1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# --- Main Execution ---
def main():
    train_loader, val_loader = load_data()

    model = AttCMAlex(num_classes=3)
    criterion = LabelSmoothingCrossEntropy(alpha=0.1, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.show()

    evaluate_model(model, val_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'attcm_alex_model.pth')


if __name__ == "__main__":
    main()
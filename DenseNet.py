import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(images_dir)
        self.masks = os.listdir(masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Mask should be single channel

        if self.transform:
            # Apply transform to the image (including normalization)
            image = self.transform(image)
            # Apply only ToTensor and Resize to the mask
            mask = transforms.ToTensor()(mask)
            mask = transforms.Resize((128, 128))(mask)  # Resize mask to 128x128

        # Ensure mask has the correct shape [1, 128, 128]
        return image, mask

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = PolypDataset(images_dir=r'/content/drive/MyDrive/ETIS-LaribPolypDB/images', masks_dir=r'/content/drive/MyDrive/ETIS-LaribPolypDB/masks', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class DenseNetSegmentationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(DenseNetSegmentationModel, self).__init__()
        self.densenet = models.densenet121(weights='IMAGENET1K_V1')  # DenseNet121

        # Adjust the first convolution layer to accept 6 channels
        self.pre_conv = nn.Conv2d(6, 3, kernel_size=1)  # Reduce 6 channels to 3

        # Convolution layers
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv2d(64, num_classes, kernel_size=1)  # Giriş kanal sayısını 64 olarak güncelle

        # Transposed convolution layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Çıkış kanal sayısını 64 olarak ayarla

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x_original, x_masked):
        # Concatenate original and masked images along the channel dimension
        x = torch.cat((x_original, x_masked), dim=1)  # Shape: [batch_size, 6, height, width]

        # Reduce channels to 3
        x = self.pre_conv(x)  # Shape: [batch_size, 3, height, width]

        # Forward pass through DenseNet
        x = self.densenet.features(x)  # Shape: [batch_size, 1024, height, width]

        # Apply adaptive pooling
        x = self.adaptive_pool(x)  # Shape: [batch_size, 1024, 16, 16]

        # Apply convolution layers
        x = F.relu(self.conv1(x))  # Shape: [batch_size, 512, 16, 16]
        x = self.upconv1(x)  # Shape: [batch_size, 256, 32, 32]
        x = self.upconv2(x)  # Shape: [batch_size, 128, 64, 64]
        x = self.upconv3(x)  # Shape: [batch_size, 64, 128, 128]

        # Final convolution
        x = self.conv4(x)  # Shape: [batch_size, num_classes, 128, 128]

        return x

# Instantiate model, loss, and optimizer
model = DenseNetSegmentationModel(num_classes=1)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def calculate_metrics(y_true, y_pred):
    # Detach y_pred from the computation graph and convert to numpy
    y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()  # Apply sigmoid and detach
    y_pred = (y_pred > 0.5).astype(np.uint8)  # Threshold to get binary prediction

    # Convert y_true to numpy
    y_true = y_true.detach().cpu().numpy()

    # Ensure y_true and y_pred are binary (0 or 1)
    y_true = (y_true > 0.5).astype(np.uint8)

    # Flatten the arrays for metric calculation
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    epoch_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()

            # Ensure masks have the correct shape [batch_size, 1, 128, 128]
            if masks.dim() == 5:  # If masks have shape [batch_size, 1, 1, 128, 128]
                masks = masks.squeeze(1)  # Remove the extra dimension

            optimizer.zero_grad()
            outputs = model(images, images)  # Use original images for both inputs

            # Ensure masks have the same size as outputs
            masks = F.interpolate(masks, size=(128, 128), mode='bilinear', align_corners=False)

            # Calculate loss
            loss = criterion(outputs, masks.float())  # masks is already [batch_size, 1, 128, 128]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate metrics
            accuracy, precision, recall, f1 = calculate_metrics(masks, outputs)
            all_preds.append(accuracy)
            all_labels.append(f1)

        # Log metrics for the epoch
        epoch_metrics['accuracy'].append(np.mean(all_preds))
        epoch_metrics['precision'].append(np.mean(all_labels))
        epoch_metrics['recall'].append(np.mean(all_preds))
        epoch_metrics['f1'].append(np.mean(all_labels))

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Plot metrics
        plot_metrics(epoch_metrics)

    return model

# Function to plot metrics
def plot_metrics(epoch_metrics):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_metrics['accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')

    plt.subplot(2, 2, 2)
    plt.plot(epoch_metrics['precision'], label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision over Epochs')

    plt.subplot(2, 2, 3)
    plt.plot(epoch_metrics['recall'], label='Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall over Epochs')

    plt.subplot(2, 2, 4)
    plt.plot(epoch_metrics['f1'], label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')

    plt.tight_layout()
    plt.show()

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(model, train_loader, criterion, optimizer, num_epochs=20)

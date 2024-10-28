import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from app.utils.unetmilesial.unet_model import UNet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.png', '_mask.png')

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()

        return image, mask

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Directories
train_images_dir = 'app/storage/trainingDataset'
train_masks_dir = 'app/storage/labels'
val_images_dir = 'app/storage/validationDataset'
val_masks_dir = 'app/storage/labels'

# Datasets and DataLoaders
batch_size = 8
train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=transform)
val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model, Loss, and Optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10 #10 training epochs is enough

# hyperparameter settings
print("\nHyperparameters:")
print(f" - Number of epochs: {num_epochs}")
print(f" - Batch size: {batch_size}")
print(f" - Learning rate: {learning_rate}")
print(f" - Optimizer: {optimizer}")
print(f" - Loss function: {criterion}")
print(f" - Model architecture:\n{model}")

# Lists to track loss values
train_losses = []
val_losses = []

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.squeeze(1)
        masks = masks.view_as(outputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f" - Training Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)
            masks = masks.view_as(outputs)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f" - Validation Loss: {val_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f'unet_epoch_{epoch+1}.pth')

# Save final model
torch.save(model.state_dict(), 'unet_final.pth')
print("\nTraining complete. Final model saved as 'unet_final.pth'.")

# Evaluation Metrics
def compute_iou(outputs, masks, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    intersection = (preds * masks).sum(dim=(1,2))
    union = (preds + masks).sum(dim=(1,2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

def compute_pixel_accuracy(outputs, masks, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    correct = (preds == masks).float().sum(dim=(1,2))
    total = preds.size(1) * preds.size(2)
    pa = (correct + 1e-6) / (total + 1e-6)
    return pa

# Evaluate on Validation Set
model.load_state_dict(torch.load('unet_final.pth'))
model.eval()
ious = []
pixel_accuracies = []

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device).squeeze(1)

        outputs = model(images)
        outputs = outputs.squeeze(1)
        masks = masks.view_as(outputs)

        batch_iou = compute_iou(outputs, masks)
        batch_pa = compute_pixel_accuracy(outputs, masks)

        ious.extend(batch_iou.cpu().numpy())
        pixel_accuracies.extend(batch_pa.cpu().numpy())

# Compute mean IoU and mean Pixel Accuracy
mean_iou = np.mean(ious)
mean_pa = np.mean(pixel_accuracies)

print("\nValidation Set Evaluation:")
print(f" - Mean IoU (mIoU): {mean_iou:.4f}")
print(f" - Mean Pixel Accuracy (mPA): {mean_pa:.4f}")

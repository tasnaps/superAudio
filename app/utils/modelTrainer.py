import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from app.utils.audio_utils import convert_audio_to_spectogram
from torch.utils.data import DataLoader
import torch.optim as optim
from skimage.transform import resize
from app.utils.unetmilesial import UNet


# Parameters for the UNet model and the DataLoader
n_channels = 1 # number of input channels
n_classes =  1# number of output classes
bilinear = True  # use bilinear interpolation for upsampling
batch_size = 16  # batch size for the DataLoader

# Path for audio files and saving wandb
low_quality_audio_dir = "app/storage/LowQualityAudios"
high_quality_audio_dir = "app/storage/HighQualityAudios"
spectrogram_dir = "D:/spectrograms"
os.makedirs(spectrogram_dir, exist_ok=True)

# Parameters for the spectrogram transform (you should adjust these values depending on your dataset)
output_size = (128, 128)

# Instantiate the UNet model and move it to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels, n_classes, bilinear).to(device)

# MSE error for the loss
criterion = nn.MSELoss()
# Adam optimizer, you may need to finetune the learning rate depending upon your task
optimizer = optim.Adam(model.parameters(), lr=0.001)


class YourSpectrogramDataset(Dataset):
    def __init__(self, low_quality_audio_dir, high_quality_audio_dir, transform=None):
        self.low_quality_audio_dir = low_quality_audio_dir
        self.high_quality_audio_dir = high_quality_audio_dir
        self.transform = transform

        self.files_dict = {}
        self.pairs = []
        hq_audio_files = [f.split(".")[0] for f in os.listdir(high_quality_audio_dir) if f.endswith('.flac')]

        for hq_audio in hq_audio_files:
            lq_audios = [f for f in os.listdir(low_quality_audio_dir) if
                         f.startswith(f"{hq_audio}-") and f.endswith('.wav')]
            self.files_dict[hq_audio] = lq_audios
            for lq_audio in lq_audios:
                self.pairs.append((hq_audio, lq_audio))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Ensure idx is within range
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError("Index out of range")

        hq_audio_file, lq_audio_file = self.pairs[idx]

        hq_audio_path = os.path.join(self.high_quality_audio_dir, f"{hq_audio_file}.flac")
        lq_audio_path = os.path.join(self.low_quality_audio_dir, lq_audio_file)

        high_quality_spectrogram = self.process_audio_file(hq_audio_path)
        low_quality_spectrogram = self.process_audio_file(lq_audio_path)

        if high_quality_spectrogram is None or low_quality_spectrogram is None:
            raise ValueError(f"None spectrogram found for index {idx} (HQ: {hq_audio_path}, LQ: {lq_audio_path})")

        #resize
        high_quality_spectrogram_np = high_quality_spectrogram.numpy()
        low_quality_spectrogram_np = low_quality_spectrogram.numpy()

        high_quality_spectrogram = resize(high_quality_spectrogram_np[0], (128, 128))
        low_quality_spectrogram = resize(low_quality_spectrogram_np[0], (128, 128))

        # Convert resized numpy arrays back to tensors and add channel dimension
        high_quality_spectrogram = torch.from_numpy(high_quality_spectrogram).unsqueeze(0).float()
        low_quality_spectrogram = torch.from_numpy(low_quality_spectrogram).unsqueeze(0).float()
        print("high and low quality spectrogram shapes")
        print(high_quality_spectrogram.shape)
        print(low_quality_spectrogram.shape)

        return low_quality_spectrogram, high_quality_spectrogram

    def process_audio_file(self, audio_path):
        file_id = os.path.basename(audio_path).replace(".wav", "").replace(".flac", "")
        spectrogram_path = os.path.join("D:/spectrograms", f"{file_id}.npy")
        if os.path.exists(spectrogram_path):
            spectrogram = np.load(spectrogram_path)
        else:
            spectrogram = convert_audio_to_spectogram(audio_path)
            if spectrogram is None:
                return None
            if self.transform:
                spectrogram = self.transform(spectrogram)
            np.save(spectrogram_path, spectrogram)

        print("Converting spectrogram to PyTorch Tensor")
        spectrogram_tensor = torch.from_numpy(spectrogram).float()
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
        print("Processing audio files function - printing spectrogram tensor shape")
        print(spectrogram_tensor.shape)
        assert spectrogram_tensor is not None, f"Failed processing {audio_path}"
        return spectrogram_tensor

    def process_audio_file(self, audio_path):
        # Use the path to generate a unique ID for the spectrogram
        file_id = os.path.basename(audio_path).replace(".wav", "").replace(".flac", "")
        spectrogram_path = os.path.join("D:/spectrograms", f"{file_id}.npy")
        print(f"Processing audio from {audio_path}")
        # Check if the spectrogram already exists
        if os.path.exists(spectrogram_path):
            print(f"Loading existing spectrogram from {spectrogram_path}")
            spectrogram = np.load(spectrogram_path)  # load the spectrogram
        else:
            # If not, create the spectrogram
            print(f"Creating new spectrogram from {audio_path}")
            spectrogram = convert_audio_to_spectogram(audio_path)
            if self.transform:
                print("Applying transform")
                spectrogram = self.transform(spectrogram)
            # Save it for future use
            print(f"Spectrogram saved at {spectrogram_path}")
            np.save(spectrogram_path, spectrogram)

        print("Converting spectrogram to PyTorch Tensor")
        spectrogram_tensor = torch.from_numpy(spectrogram).float()  # Convert spectrogram to PyTorch tensor
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0)  # PyTorch expects the channel dimension first
        assert spectrogram_tensor is not None, f"Failed processing {audio_path}"
        return spectrogram_tensor

class SpectogramTransform:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, spectrogram):
        # Resize
        spectrogram = resize(spectrogram, self.output_size)
        min_val = np.min(spectrogram)
        max_val = np.max(spectrogram)
        spectrogram = (spectrogram - min_val) / (max_val - min_val)
        return spectrogram

transform = SpectogramTransform(output_size)

spectrogram_dataset = YourSpectrogramDataset(
    low_quality_audio_dir='/app/storage/ProcessedAudios/LowQualityAudios',
    high_quality_audio_dir='/app/storage/ProcessedAudios/HighQualityAudios',
    transform=transform
)

# DataLoader for batching and shuffling
spectrogram_dataloader = DataLoader(spectrogram_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (low_quality, high_quality) in enumerate(spectrogram_dataloader):
        low_quality = low_quality.to(device)
        high_quality = high_quality.to(device)

        # Forward pass
        outputs = model(low_quality)

        # Compute loss
        loss = criterion(outputs, high_quality)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Reset gradients
        optimizer.zero_grad()

        running_loss += loss.item()

    running_loss /= len(spectrogram_dataloader)
    print(f"Epoch {epoch + 1}, Training Loss: {running_loss:.6f}")

    # Save model state at the end of each epoch
    torch.save(model.state_dict(), f'unet_epoch_{epoch}.pth')
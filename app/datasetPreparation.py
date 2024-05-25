import os
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from utils.audio_utils import convert_audio_to_spectogram
from torch.utils.data import DataLoader
import torch.optim as optim
from skimage.transform import resize
from models.enhancement_model import UNet
low_quality_audio_dir = "app/storage/processed_audios"
high_quality_audio_dir = "app/storage/HighQualityAudios"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# MSE error for the loss
criterion = nn.MSELoss()
# Adam optimizer, you may need to finetune the learning rate depending upon your task
optimizer = optim.Adam(model.parameters(), lr=0.001)
spectrogram_dir = "D:/spectrograms"
os.makedirs(spectrogram_dir, exist_ok=True)

# Parameters for the spectrogram transform (you should adjust these values depending on your dataset)
output_size = (128, 128)


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

        # Normalize (using Min-Max scaling)
        min_val = np.min(spectrogram)
        max_val = np.max(spectrogram)
        spectrogram = (spectrogram - min_val) / (max_val - min_val)

        return spectrogram

# Initialize the transform
transform = SpectogramTransform(output_size)

# You can also add a transform to convert the amplitude spectrogram to a dB scale.
# If your model expects a dB scale spectrogram, this can be part of the transform.
spectrogram_dataset = YourSpectrogramDataset(
    low_quality_audio_dir='C:/Users/tapio/PycharmProjects/superAudio/app/storage/processed_audios',
    high_quality_audio_dir='C:/Users/tapio/PycharmProjects/superAudio/app/storage/HighQualityAudios',
    transform=transform
)


#dataloader for batching and shuffling
spectrogram_dataloader = DataLoader(spectrogram_dataset, batch_size=16, shuffle=True, num_workers=0)
optimizer.zero_grad()

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (low_quality, high_quality) in enumerate(spectrogram_dataloader):
        low_quality = low_quality.to(device)
        high_quality = high_quality.to(device)

        outputs = model(low_quality)

        loss = criterion(outputs, high_quality)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    running_loss /= len(spectrogram_dataloader)
    print(f"Epoch {epoch + 1}, Training Loss: {running_loss:.6f}")
    torch.save(model.state_dict(), f'unet_epoch_{epoch}.pth')
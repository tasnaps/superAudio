import os
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
high_quality_audio_dir = "app/storage/HighQuality_audios"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# MSE error for the loss
criterion = nn.MSELoss()
# Adam optimizer, you may need to finetune the learning rate depending upon your task
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Parameters for the spectrogram transform (you should adjust these values depending on your dataset)
output_size = (128, 128)

class YourSpectrogramDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        """
        Your custom dataset initialization.
        :param audio_dir: Directory with all the audio files.
        :param transform: Optional transform to be applied on a sample.
        """
        self.low_quality_audio_dir = low_quality_audio_dir
        self.high_quality_audio_dir = high_quality_audio_dir
        self.audio_files = [f for f in os.listdir(low_quality_audio_dir) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        low_quality_audio_path = os.path.join(self.low_quality_audio_dir, self.audio_files[idx])
        high_quality_audio_path = os.path.join(self.high_quality_audio_dir, self.audio_files[idx])

        # Convert both audio files to spectrogram
        low_quality_spectrogram = convert_audio_to_spectogram(low_quality_audio_path)
        high_quality_spectrogram = convert_audio_to_spectogram(high_quality_audio_path)

        # Apply any processing or transformations you might want here
        if self.transform:
            low_quality_spectrogram = self.transform(low_quality_spectrogram)
            high_quality_spectrogram = self.transform(high_quality_spectrogram)

        # Convert spectrograms to PyTorch tensors
        low_quality_spectrogram_tensor = torch.from_numpy(low_quality_spectrogram).float()
        high_quality_spectrogram_tensor = torch.from_numpy(high_quality_spectrogram).float()

        # PyTorch expects the channel dimension first, so use unsqueeze to add it
        low_quality_spectrogram_tensor = low_quality_spectrogram_tensor.unsqueeze(0)
        high_quality_spectrogram_tensor = high_quality_spectrogram_tensor.unsqueeze(0)

        return low_quality_spectrogram_tensor, high_quality_spectrogram_tensor

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
    low_quality_audio_dir='storage/processed_audios/low_quality',
    high_quality_audio_dir='storage/processed_audios/high_quality',
    transform=transform
)


#dataloader for batching and shuffling
spectrogram_dataloader = DataLoader(spectrogram_dataset, batch_size=16, shuffle=True)

for i, (low_quality, high_quality) in enumerate(spectrogram_dataloader):
    # move data to GPU if available
    low_quality = low_quality.to(device)
    high_quality = high_quality.to(device)

    # forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(low_quality)

    # calculate the loss
    loss = criterion(outputs, high_quality)

    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # perform a single optimization step (parameter update)
    optimizer.step()

    # update running training loss
    # print training statistics
    # print every 20 iterations
    if i % 20 == 0:
        print('Iteration %d, loss %.6f' % (i, loss.item()))

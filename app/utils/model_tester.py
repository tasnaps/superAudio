from app.utils.unetmilesial import UNet
from torch.utils.data import DataLoader, Dataset
from app.utils.audio_utils import convert_audio_to_spectogram, preprocess_audio, convert_spectrogram_to_audio
import numpy as np
import torch
import soundfile as sf


class SpectrogramTestDataset(Dataset):
    def __init__(self, spectrogram):
        self.spectrogram = spectrogram

    def __len__(self):
        return self.spectrogram.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.spectrogram[idx]


# Check if a cuda-capable GPU is available and if so use it, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_channels = 1  # number of input channels
n_classes = 1  # number of output classes
bilinear = True  # use bilinear interpolation for upsampling

model = UNet(n_channels, n_classes, bilinear).to(device)

# Report the model's architecture
print(model)

song = "C:/Users/tapio/PycharmProjects/superAudio/app/storage/RawAudios/EvaluationSet/(01) [Fragma] Toca's Miracle.mp3"
song = preprocess_audio(song)
spectrogram_raw = convert_audio_to_spectogram(song)

# Calculate number of frames for 10 seconds
hop_length = 512  # Default used by librosa.stft() when not specified
sample_rate = 48000
num_frames_10_sec = sample_rate * 180 // hop_length
spectrogram = spectrogram_raw[:, :num_frames_10_sec]

weights = 'C:/Users/tapio/PycharmProjects/superAudio/app/models/Model2/unet_epoch_9.pth'
model.load_state_dict(torch.load(weights, map_location=device))  # load weights to the right device
model.eval()

# Then convert the input to a PyTorch Tensor and copy it to GPU
spectrogram_tensor = torch.from_numpy(spectrogram)
spectrogram_tensor = spectrogram_tensor.unsqueeze(0).unsqueeze(0)
spectrogram_tensor = spectrogram_tensor.to(device)

# Spectrogram for inference
spectrogram_for_inference = SpectrogramTestDataset(spectrogram_tensor)

# DataLoader for batching
testloader = DataLoader(spectrogram_for_inference, batch_size=1, shuffle=False, num_workers=0)

enhanced_spectrograms = []

with torch.no_grad():
    for data in testloader:
        spectrogram_batch = data.to(device)
        output = model(spectrogram_batch)
        # Note: Converting the Tensor right after model output back to the CPU
        # for conversion to numpy and subsequent operations
        enhanced_spectrogram = output.squeeze(0).cpu().numpy()
        enhanced_spectrograms.append(enhanced_spectrogram)

# Concatenate all enhanced spectrogram batches:
enhanced_spectrogram = np.concatenate(enhanced_spectrograms, axis=0)

# Convert the enhanced spectrogram back into an audio signal
enhanced_audio = convert_spectrogram_to_audio(enhanced_spectrogram)

# Save the enhanced audio
enhanced_audio_path = "C:/Users/tapio/PycharmProjects/superAudio/app/storage/EnhancedAudios/enhancedModel2.wav"
print(type(enhanced_audio), enhanced_audio.dtype, enhanced_audio.shape)
enhanced_audio = enhanced_audio.reshape(-1)
sf.write(enhanced_audio_path, enhanced_audio, 48000)

print(f'Enhanced audio saved at {enhanced_audio_path}')

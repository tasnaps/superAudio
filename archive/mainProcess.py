from models.preprocessing import process_all_audio_files
from utils.audio_utils import convert_audio_to_spectogram
from models.enhancement_model import UNet
import torch
import os
import numpy as np

# Define your paths
input_directory = 'storage/HighQualityAudios'
output_directory = 'storage/processed_audios'
model_save_path = 'models/unet.pth'

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Assume you have a dataset and dataloader set up
# This is a placeholder for your dataset which you would need to implement
dataset = YourSpectrogramDataset(output_directory)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop - This is a simplified version. You'll need to add more details
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        predictions = model(inputs)

        # Compute loss
        loss = loss_function(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)


# Function to convert and process all audio files
def process_all_audio_files(input_directory, output_directory):
    # Iterate through all files in the input directory
    for file_name in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file_name)

        # Check if it's an audio file
        if file_path.lower().endswith('.wav'):
            # Preprocess the audio (denoise, resample, etc.)
            preprocessed_audio_path = preprocess_audio(file_path)

            # Convert to spectrogram
            spectrogram = convert_audio_to_spectrogram(preprocessed_audio_path)

            # Save the spectrogram
            # You might want to use numpy to save as .npy file or imageio for an image file
            output_spectrogram_path = os.path.join(output_directory, os.path.splitext(file_name)[0] + '.npy')
            np.save(output_spectrogram_path, spectrogram)


if __name__ == "__main__":
    # Run the audio processing and spectrogram conversion
    process_all_audio_files(input_directory, output_directory)

    # Further code to handle model training would go here

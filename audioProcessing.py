from app.utils import audio_utils
import os
import soundfile as sf
import librosa
import random

hq_audio_directory = 'app/storage/HighQualityAudios/'
processed_audio_path = 'app/storage/LowQualityAudios/'
if not os.path.isdir(hq_audio_directory):
    print(f"Error: {hq_audio_directory} directory does not exist.")
    exit(1)

if not os.path.isdir(processed_audio_path):
    print(f"Error: {processed_audio_path} directory does not exist.")
    exit(1)

print("Directories found")


if __name__ == '__main__':
    # Go through all items in directory to make them shittier
    for filename in os.listdir(hq_audio_directory):
        if filename.endswith('.flac'):
            file_path = os.path.join(hq_audio_directory, filename)
            # Load audio
            audio, sample_rate = librosa.load(file_path)
            # Downsample
            downsampled_audio = audio_utils.downsampler_two(audio, sample_rate, 48000)

            # Add noise and low pass filter it with random variations
            for i in range(3):
                # Random noise factor between 0 and 1
                noise_factor = random.random() * 0.4
                noised_audio = audio_utils.add_noise(downsampled_audio, noise_factor)

                # Random cut off frequency between 200 and 9000
                cutoff_freq = random.randint(200, 9000)
                filtered_audio = audio_utils.apply_low_pass_filter(noised_audio, 48000, cutoff_freq)

                # Name our baby with noise and frequency details for clarity
                new_filename = f"{os.path.splitext(filename)[0]}_processed_{i}_noise_{noise_factor}_freq_{cutoff_freq}.wav"
                sf.write(os.path.join(processed_audio_path, new_filename), filtered_audio, 48000)

    print("Processing done for all files in directory.")
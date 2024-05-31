from app.utils.audio_utils import convert_audio_to_spectogram
import os

#Initiate spectogram creation

def process_all_audio_files(input_directory, output_directory):
    #Todo, choose what type of file model needs and save as that
    for file_name in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file_name)
        if file_path.lower().endswith('.wav'):  # Check if it's an audio file
            spectrogram = convert_audio_to_spectogram(file_path)
            # Here, implement logic to save the spectrogram array to disk,
            # such as in a NumPy file or image, as needed for your model
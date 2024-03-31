import numpy as np
from scipy.io import wavfile
import scipy.signal
from scipy.signal import butter, lfilter
import librosa
import soundfile as sf

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)

    #placeholder for denoiser
    #todo denoising algorithm
    denoised_audio = denoise_audio(audio)

    #match bitrate

    if sr != 16000:
        resampled_audio = librosa.resample(denoised_audio, orig_sr=sr, target_sr=16000)
    else:
        resampled_audio = denoised_audio

    #save audio to a new file
    output_path = "../storage/preprocessed_audios/preprocessed_audio.wav"
    sf.write(output_path, resampled_audio, 16000)

    return output_path
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def denoise_audio(file_name, lowcut, highcut):
    fs, data = wavfile.read(file_name)
    y = butter_bandpass_filter(data, lowcut, highcut, fs)
    return y
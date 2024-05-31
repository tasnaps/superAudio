import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import soundfile as sf


# Function to convert audio into spectogram.
def convert_audio_to_spectogram(file_path, save_path=None):
    """
    Converts audio file into a spectogram.
    :param file_path: The path of the audio file to be converted.
    :return: The spectrogram of the audio file as a NumPy array.
    """
    print(f"Loading audio from {file_path}")
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=None)
    print(f"Converting audio to spectrogram with sample_rate: {sample_rate}")
    # Convert the audio to a spectogram
    spectogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

    if not np.any(audio):
        print(f"Warning: Loaded audio from {file_path} contains only zero values.")
    print(
        f"Audio converted to spectrogram, shape: {spectogram.shape}, min: {np.min(spectogram)}, max: {np.max(spectogram)}")

    #option to save the spectogram to a file
    if save_path:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectogram, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-frequency power spectrogram')
        print(f"Saving spectrogram image at: {save_path}")
        plt.savefig(save_path)
        plt.close()

    return spectogram

def downsampler_two(audio, original_rate, target_rate):
    """
        Downsamples the given audio to the target sample rate using Scipy.

        :param audio: The audio data to be downsampled.
        :param original_rate: The original sample rate of the audio.
        :param target_rate: The target sample rate for downsampling.
        :return: The downsampled audio.

        """
    resample_ratio = target_rate / original_rate
    new_audio_len = int(len(audio) * resample_ratio)

    downsampled_audio = resample(audio, new_audio_len)

    return downsampled_audio
def add_noise(audio, noise_factor):
    """
    Adds random noise to an audio signal.

    :param audio: The original audio signal.
    :type audio: numpy.ndarray

    :param noise_factor: The scaling factor for the noise.
    :type noise_factor: float

    :return: The audio signal with added noise.
    :rtype: numpy.ndarray

    """
    noise = np.random.randn(len(audio))
    audio_noisy = audio + noise_factor * noise
    audio_noisy = audio_noisy.astype(type(audio[0]))

    return audio_noisy


def apply_low_pass_filter(audio, sample_rate, cutoff_hz):
    """
    Apply a low-pass filter to the audio signal.

    :param audio: The input audio signal, represented as an array-like object.
    :param sample_rate: The sample rate of the audio signal.
    :param cutoff_hz: The cutoff frequency of the low-pass filter, in hertz.
    :return: The filtered audio signal.
    """
    nyquist_rate = sample_rate / 2.
    normalized_cutoff = cutoff_hz / nyquist_rate
    b, a = scipy.signal.butter(1, normalized_cutoff, btype='low')
    audio_filtered = scipy.signal.lfilter(b, a, audio)

    return audio_filtered


def preprocess_audio(file_path):
    #Todo, might be an issue here in resampling audio, check functionality, if not working use audio_utils resampling instead
    audio, sr = librosa.load(file_path, sr=None)

    #placeholder for denoiser
    #todo denoising algorithm
    denoised_audio = denoise_audio(audio)

    #match bitrate

    if sr != 48000:
        resampled_audio = librosa.resample(denoised_audio, orig_sr=sr, target_sr=48000)
    else:
        resampled_audio = denoised_audio

    #save audio to a new file
    output_path = "../storage/ProcessedAudios/LowQualityAudios/preprocessed_audio.wav"
    sf.write(output_path, resampled_audio, 48000)

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
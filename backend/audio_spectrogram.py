
import librosa
import numpy as np
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_db = np.expand_dims(spectrogram_db, axis=-1)  # Add channel dimension
    return spectrogram_db
import librosa
import numpy as np
import cv2
def process_audio(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # Generate spectrogram (e.g., Mel-spectrogram)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to log scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Ensure the shape is (128, 5168, 1)
    spectrogram = np.expand_dims(S_db, axis=-1)  # Shape: (128, 5168, 1)

    # Replicate the single channel to make it 3 channels
    spectrogram_3d = np.repeat(spectrogram, 3, axis=-1)  # Shape: (128, 5168, 3)

    # Resize to (128, 128, 3) as expected by your model
    spectrogram_resized = cv2.resize(spectrogram_3d, (128, 128))  # Resize to (128, 128, 3)

    # Add batch dimension (1, 128, 128, 3)
    spectrogram_resized = np.expand_dims(spectrogram_resized, axis=0)

    return spectrogram_resized

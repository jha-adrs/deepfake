from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
import cv2
import os
from moviepy.editor import VideoFileClip

app = Flask(__name__)
audio_model_path = "models/deepfake_video.keras"
video_model_path = "models/deepfake_audio.h5"

video_model = tf.keras.models.load_model(video_model_path)
audio_model = tf.keras.models.load_model(audio_model_path)

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_db = np.expand_dims(spectrogram_db, axis=-1)  # Add channel dimension
    return spectrogram_db

# Process video and split into frames
def process_video(video_path):
    video_frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize each frame to the required shape (224, 224, 3)
        frame_resized = cv2.resize(frame, (224, 224))
        video_frames.append(frame_resized)

    cap.release()
    video_frames = np.array(video_frames)  # Shape: (num_frames, 224, 224, 3)
    return video_frames

# Split audio and video from uploaded file
def split_audio_video(video_path):
    video_clip = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_path)
    return audio_path, video_path

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

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    audio_file = request.files.get('audio')
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)
    
    # Process audio and generate spectrogram
    spectrogram = process_audio(audio_path)
    
    # Make prediction with the model
    prediction = audio_model.predict(spectrogram)
    
    # Remove the temporary audio file
    #os.remove(audio_path)
    
    return jsonify({
        "audio_prediction": prediction.tolist(),
        })

# Endpoint for video inference
@app.route('/predict_video', methods=['POST'])
def predict_video():
    video_file = request.files.get('video')
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    # Process video to get frames
    video_frames = process_video(video_path)

    # Prepare to collect predictions for each frame
    frame_predictions = []
    for frame in video_frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 224, 224, 3)
        prediction = video_model.predict(frame)
        frame_predictions.append(prediction)

    # Average predictions across all frames
    avg_prediction = np.mean(frame_predictions, axis=0)
    #print(frame_predictions)
    os.remove(video_path)
    return jsonify({
        "video_prediction": avg_prediction.tolist()
        })

# Endpoint for combined inference
@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    video_file = request.files.get('video')
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    # Extract audio from the video
    video_clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_path)

    # Process audio and make prediction
    audio_spectrogram = process_audio(audio_path)
    audio_prediction = audio_model.predict(audio_spectrogram)
     # Process video frames and make predictions
    video_frames = process_video(video_path)
    frame_predictions = []

    for frame in video_frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 224, 224, 3)
        frame_prediction = video_model.predict(frame)
        frame_predictions.append(frame_prediction)

    # Average predictions across all frames
    avg_video_prediction = np.mean(frame_predictions, axis=0)

    # Clean up temporary files
    #os.remove(video_path)
    #os.remove(audio_path)
    # Return the predictions for both audio and video
    return jsonify({
        "audio_prediction": audio_prediction.tolist(),
        "video_prediction": avg_video_prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
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

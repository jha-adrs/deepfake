def split_audio_video(video_path):
    video_clip = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_path)
    return audio_path, video_path

from moviepy.editor import VideoFileClip

video = VideoFileClip("voice/audio/1.mp4")
audio = video.audio

audio.write_audiofile("voice/audio/1.mp3")

video.close()
audio.close()

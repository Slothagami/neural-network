# tools for working with audio
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import AudioFileClip

def save_audio(audio, filen):
    audio = audio.reshape(audio.shape[1], 1)
    audio = AudioArrayClip(audio, fps=44100/2)
    audio.write_audiofile(f"{filen}.mp3")

def load_audio(path):
    audio = AudioFileClip(path)
    return audio.to_soundarray()

if __name__ == "__main__":
    wave = load_audio("voice/audio/11MM_18E Extrema Problems_downsample.mp3")
    print(wave.max(), wave.min())

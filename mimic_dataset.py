from moviepy.editor import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np 
import matplotlib.pyplot as plt

def save_sample(sample):
    c = cuts[sample]
    c = c.reshape(c.shape[0], 1)
    audio = AudioArrayClip(c, fps=44100/2)
    audio.write_audiofile(f"{lines[sample]}.mp3")

def audio_frame(timestamp):
    min, sec = [int(x) for x in timestamp.split(":")]
    sec += 60 * min # time in seconds
    return sec * fps

def section_times(transcript, sample_rate):
    with open(transcript, "r") as file:
        # transform lines to instructions to cut file
        last_cut = 0
        timestamp = True
        cuts = []
        lines = []
        for line in file.readlines():
            line = line.strip()
            if timestamp:
                # Calculate index of audio for the timestamp
                if line == "00:00": 
                    timestamp = False
                    continue

                cut = audio_frame(line)
                cuts.append(wave[last_cut:cut])
                last_cut = cut
            else:
                lines.append(line)

            timestamp = not timestamp

    return cuts, lines

# Cut the audio file into sectoins acording to the transcript
audio = AudioFileClip("voice/audio/11MM_18E Extrema Problems_downsample.mp3")

print("Creating Training Data...")
downsample = 4
wave = audio.to_soundarray()[:,0][::downsample]

# noise_strength = .05
# noise = (np.random.rand(*wave.shape) - .5) * 2 * noise_strength

# label_sample = wave 
# train_sample = (label_sample + noise) / (1 + noise_strength)

fps = audio.fps // downsample
sample_duration = 6 # seconds
sample_size = fps * sample_duration

cuts, lines = section_times("voice/transcripts/11MM_18E Extrema Problems.txt", audio.fps)
# print(cuts[:4])
# print(" ".join(lines))

save_sample(32)

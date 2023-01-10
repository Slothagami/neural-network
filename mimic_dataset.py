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

def section_times(transcript):
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

                cut_frame = audio_frame(line)
                cut = wave[last_cut:cut_frame]

                # padd the recording to match the input of the Network
                clip_length = cut.shape[0]//fps # in seconds

                if clip_length <= sample_duration: # ignore clips too long to input to nn
                    cut = np.pad(cut, (0, sample_size - cut.shape[0]))

                    cuts.append(cut)
                last_cut = cut_frame
            else:
                lines.append(line)

            timestamp = not timestamp

    return cuts, lines

def make_data():
    global fps, sample_duration, sample_size, cuts, lines, wave
    print("Creating Training Data...")
    # Cut the audio file into sectoins acording to the transcript
    audio = AudioFileClip("voice/audio/11MM_18E Extrema Problems_downsample.mp3")

    downsample = 4
    wave = audio.to_soundarray()[:,0][::downsample]

    # noise_strength = .05
    # noise = (np.random.rand(*wave.shape) - .5) * 2 * noise_strength

    # label_sample = wave 
    # train_sample = (label_sample + noise) / (1 + noise_strength)

    fps = audio.fps // downsample
    sample_duration = 10 # seconds
    sample_size = fps * sample_duration

    cuts, lines = section_times("voice/transcripts/11MM_18E Extrema Problems.txt")
    return cuts, lines
    # save_sample(0)
    # print(np.array(cuts).shape)


if __name__ == "__main__": make_data()

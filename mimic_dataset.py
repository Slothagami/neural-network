from moviepy.editor import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np 
import matplotlib.pyplot as plt

def save_sample(sample,arr):
    c = arr[sample]
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
        remove_next_line = False
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
                else:
                    # remove corresponding subtitle
                    remove_next_line = True

                last_cut = cut_frame
            else:
                if not remove_next_line:
                    lines.append(line)
                remove_next_line = False

            timestamp = not timestamp

    return cuts, lines

def collect_samples():
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
    return np.array(cuts), lines

def encode_subtitles(subs):
    characters = " abcdefghijklmnopqrstuvwxyz0123456789'=-+*^/?"
    lines = []
    for subtitle in subs:
        subtitle = subtitle.ljust(45, " ")
        subtitle = np.array([characters.index(c)/len(characters) for c in subtitle])
        lines.append(subtitle)

    return np.array(lines)

def make_data():
    samples, subitles = collect_samples()

    # assemble training data
    labels = np.copy(samples)

    # add noise to input
    noise_ammounts = np.random.rand(len(samples))
    noise = np.random.rand(*samples.shape) * noise_ammounts[:, np.newaxis] # multiply each column with the strength of noise
    samples += noise

    norm = (1 / (1 + noise_ammounts))
    samples = samples * norm[:, np.newaxis] # normalize input between -1 and 1

    # add noise indicator neuron to the bottom row
    # samples = np.hstack((noise_ammounts,samples))

    # encode subtitles into input (max-length: 45 characters)
    coded_subtitles = encode_subtitles(subitles)

    training_samples = []
    for audio, noise, subtitle in zip(samples, noise_ammounts, coded_subtitles):
        input = np.concatenate((audio, [noise], subtitle))
        training_samples.append( input.reshape(input.shape[0], 1) )

    return training_samples, labels



if __name__ == "__main__": 
    data = make_data()

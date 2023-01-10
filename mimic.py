from network.neuralnet   import *
from network.activations import *

from moviepy.editor import AudioFileClip
import numpy as np 
import matplotlib.pyplot as plt

# Preprocessing
print("Loading Audio...")
audio = AudioFileClip("voice/audio/11MM_18E Extrema Problems_downsample.mp3")

print("Creating Training Data...")
downsample = 4
wave = audio.to_soundarray()[:,0][::downsample]

noise_strength = .05
noise = (np.random.rand(*wave.shape) - .5) * 2 * noise_strength

label_sample = wave 
train_sample = (label_sample + noise) / (1 + noise_strength)

fps = audio.fps // downsample
sample_duration = 6 # seconds
sample_size = fps * sample_duration

# plt.plot(noise[:fps])
# plt.show()

# print(f"{sample_size:,}")

train_sample = train_sample[:fps]
label_sample = label_sample[:fps]

sample_size = fps 

# Train
nn = NeuralNet(lr=.002)
nn.config((sample_size, 100, 50, 100, sample_size), ReLU)

print("Beginning Training...")

for _ in range(20):
    train_samples = []
    for _ in range(10):
        noise  = (np.random.rand(*label_sample.shape) - .5) * 2 * noise_strength
        sample = (label_sample + noise) / (1 + noise_strength)
        train_samples.append([sample])

    nn.train(train_samples, [[label_sample]]*10, 15)
    noise_strength += .05
    print(noise_strength)

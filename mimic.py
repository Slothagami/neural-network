from network.neuralnet   import *
from network.activations import *
from mimic_dataset import make_data, encode_subtitles

from moviepy.audio.AudioClip import AudioArrayClip

def save_output_audio(audio, prompt):
    audio = audio.reshape(audio.shape[1], 1)
    audio = AudioArrayClip(audio, fps=44100/2)
    audio.write_audiofile(f"{prompt}.mp3")

samples, labels = make_data()
in_size = samples[0].shape[1]
out_size = labels[0].shape[0]

# Train
nn = NeuralNet("beaumont.npy", lr=.1)
# nn.config((in_size, 100, 50, 100, out_size), ReLU)
nn.config((in_size, 150, 80, 80, 150, out_size), Tanh)
nn.load()

# print("Beginning Training...")
# for _ in range(10):
#     nn.train(samples, labels, 4)
#     samples, labels = make_data() # Randomize the noise in the data

# Test Sample
noise_size = 11025 * 5
noise = np.random.rand(noise_size)
prompt = "welcome back everyone"
prompt_enc = encode_subtitles([prompt])[0]

input = np.concatenate((noise, [1], prompt_enc))

audio = nn.predict([input])[0]
print(audio.shape)
save_output_audio(audio, prompt)

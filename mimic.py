from network.neuralnet   import *
from network.activations import *
from network.audio       import *

from network.diffusion import *
from mimic_dataset import collect_samples

audio, subtitles = collect_samples()

# samples, labels = make_data()
# in_size = samples[0].shape[1]
# out_size = labels[0].shape[0]

sample_size = audio.shape[1] # audio length in samples

# Train
# nn = NeuralNet("beaumont.npy", lr=.1)
# # nn.config((in_size, 100, 50, 100, out_size), ReLU)
# nn.config((sample_size, 150, 80, 80, 150, sample_size), Tanh)
# # nn.load()

# print("Beginning Training...")
# n_epochs = 10
# batch_size = 1
# for epoch in range(n_epochs):
#     t = np.random.randint(0, DIFFUSION_STEPS, size=(batch_size,))
#     audio = np.stack([audio] *  batch_size)
#     audio, noise = diffuse(audio, t, betas = linear_schedule())
#     nn.train(audio, noise, 1)
    

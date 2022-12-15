from network.neuralnet import NeuralNet
from network.gan import GAN 

generator = NeuralNet()
generator.config((20*20, 100, 200, 20*20))

discriminator = NeuralNet()
discriminator.config((20*20, 100, 50, 2))

gan = GAN(generator, discriminator)

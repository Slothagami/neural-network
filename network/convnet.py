from network.neuralnet import * 

class ConvNet:
    def __init__(self, generator: NeuralNet, discriminator: NeuralNet):
        self.generator = generator
        self.discriminator = discriminator

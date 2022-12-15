from network.neuralnetwork import * 

class ConvNet:
    def __init__(self, generator: NeuralNetwork, discriminator: NeuralNetwork):
        self.generator = generator
        self.discriminator = discriminator

# Neural Network
An Neural Network Libary, made completely from scratch using numpy and python, originally based on [this article](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65), currently supports:
- loading, and automatic saving, of trained models
- Fully Connected Layers
- Convolutional Layers
- Softmax
- Sigmoid Activation
- Tanh Activation
- ReLU Activation
- Custom layer types (inherits `Layer`)
- Custom Activations and Loss functions (inherits `NNFunction`)
- Mean Squared Error loss
- Categorical Cross Entropy loss

### TODO
- update pooling layer for multiple color channels
- convert modules to C for speed?
- UNets
- Diffusion Models

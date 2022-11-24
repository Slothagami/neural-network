class Network:
    def __init__(self, /, loss, loss_prime, lr=.1):
        self.layers = []

        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer): self.layers.append(layer)

    def predict(self, samples):
        # Predict a Batch of samples
        results = []
        for sample in samples:
            output = sample

            for layer in self.layers:
                output = layer.forward(output)

            results.append(output)

        return results

    def train(self, samples, labels, epochs):
        for epoch in range(epochs):
            disp_error = 0 

            for sample, label in zip(samples, labels):
                # Forward Propogation
                output = sample 
                for layer in self.layers:
                    output = layer.forward(output)

                # Display Error
                disp_error += self.loss(label, output)

                # Backprop
                error = self.loss_prime(label, output)
                for layer in reversed(self.layers):
                    error = layer.backprop(error, self.lr)

            # Calc Average Error
            error /= len(samples)

            print(f"Epoch: {epoch}, Error: {disp_error}")

import numpy as np
import matplotlib.pyplot as plt

def show_output_space(nn, /, image_size=100):
    # Image of predictions for all values
    image = np.zeros((image_size, image_size))
    for x in range(image_size):
        for y in range(image_size):
            image[x, y] = nn.predict([x / image_size, y / image_size])

    plt.imshow(image, "Greys", vmin=0, vmax=1)
    plt.show()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DIFFUSION_STEPS = 6

def linear_schedule():
    return np.linspace(0, 1, DIFFUSION_STEPS)

def cos_schedule():
    def f(t):
        s = .008
        return np.cos( ((t/DIFFUSION_STEPS + s)/(1 + s) * np.pi/2) ) ** 2

    return np.array([1 - f(t)/f(0) for t in range(DIFFUSION_STEPS)])

def square_schedule():
    return np.linspace(0, 1, DIFFUSION_STEPS) ** 2

def pow_schedule(power):
    # using a power schedule like this would mean that the backward diffusion converges quickly? less iterations needed for a fine image?
    # also could mean the innitial amount of noise is too large to be tangible to learn
    return np.linspace(0, 1, DIFFUSION_STEPS) ** power


def diffuse(x0, t, betas = linear_schedule()):
    alphas         = 1 - betas 
    alpha_hat      = np.cumprod(alphas) # product of all previous alphas
    noise_ammounts = alpha_hat[t].reshape(-1,1,1,1)

    noise    = np.random.normal(size = x0.shape) # sample normal dist
    mean     = np.sqrt(noise_ammounts) * x0
    variance = np.sqrt(1 - noise_ammounts) * noise
    return mean + variance, noise


def format_image(img):
    # reshape image to range -1 to 1
    size = 128
    img = img.resize((size, size))
    arr = np.asarray(img)
    arr = 2 * (arr / 255) - 1
    return arr

def unformat_image(arr):
    # reshape image from -1 to 1 to 0-255
    arr = (arr + 1) / 2 # 0-1
    arr *= 255
    arr = arr.astype("uint8")
    img = Image.fromarray(arr)
    return img

if __name__ == "__main__":
    # Diffuse an image and display the process
    img = Image.open("voice/nnnotebook/chad.png").convert("RGB")
    img = format_image(img)

    t = np.array([x for x in range(DIFFUSION_STEPS)])
    input_images = np.stack([img] * len(t))
    results, _ = diffuse(input_images, t, betas=pow_schedule(4))

    # Display Diffusion process
    fig = plt.figure(figsize=(20,3))

    for n, result in enumerate(results):
        fig.add_subplot(1, len(t), n+1)
        plt.imshow(unformat_image(result))
        plt.axis("off")

    plt.show()

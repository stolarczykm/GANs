import numpy as np
from tensorflow.keras import models, layers

from gan import GAN, get_gan_package_root_direcotry
from gan.animation import ScatterAnimation

NOISE_DIM = 16


def make_generator():
    return models.Sequential([
        layers.Dense(10, input_shape=(NOISE_DIM,)),
        layers.LeakyReLU(),
        layers.Dense(10),
        layers.LeakyReLU(),
        layers.Dense(10),
        layers.LeakyReLU(),
        layers.Dense(2),
    ])


def make_discriminator():
    return models.Sequential([
        layers.Dense(10, input_shape=(2,)),
        layers.LeakyReLU(),
        layers.Dense(10),
        layers.LeakyReLU(),
        layers.Dense(10),
        layers.LeakyReLU(),
        layers.Dense(1),
    ])


def create_dataset(n):
    t = np.random.uniform(0, 1, (n,))
    X = np.c_[np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)] + np.random.normal(scale=0.1, size=(n, 2))
    return X


def _main():
    X = create_dataset(2048)
    noise = np.random.normal(0, 1, (1024, NOISE_DIM))
    generator = make_generator()
    discriminator = make_discriminator()
    gan = GAN(generator, discriminator)
    generated, gen_loss, disc_loss = gan.train(X, epochs=5000, batch_size=512, noise=noise, generate_frequency=10)
    animation = ScatterAnimation(animation_length=30000)
    animation.animate(X, generated, gen_loss, disc_loss)
    animation.save("movie2.mp4")


if __name__ == '__main__':
    _main()

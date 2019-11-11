import os
import datetime
from typing import Optional

import numpy as np
from tensorflow.keras import models, layers

from gan import GAN, get_models_directory, get_data_directory
from gan.animation import ScatterAnimation, ImageAnimation

NOISE_DIM = 128


def make_generator():
    model = models.Sequential([
        layers.Dense(7 * 7 * 256, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same"),
    ])
    assert model.output_shape == (None, 28, 28, 1)
    return model


def make_discriminator():
    return models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(28, 28, 1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),
    ])


def create_dataset(path: Optional[str] = None):
    if path is None:
        path = os.path.join(get_data_directory(), "mnist.npz")
    X = np.load(path)["x_train"][..., np.newaxis]
    X = X.astype(np.float32) / 127.5 - 0.5
    return X


if __name__ == '__main__':
    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_dir = os.path.join(get_models_directory(), run_name)
    os.makedirs(model_dir, exist_ok=False)

    X = create_dataset()
    noise = np.random.normal(0, 1, (100, NOISE_DIM))

    generator = make_generator()
    discriminator = make_discriminator()
    gan = GAN(generator, discriminator)

    generated, gen_loss, disc_loss = gan.train(
        X, epochs=50, batch_size=512, noise=noise, generate_frequency=1, save_dir=model_dir, save_frequency=1)

    animation = ImageAnimation(animation_length=5000)
    animation.animate(X, generated, gen_loss, disc_loss)
    animation.save(os.path.join(model_dir,  "animation.mp4"))






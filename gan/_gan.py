import logging
import os

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


class GAN:
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model):
        self._generator = generator
        self._discriminator = discriminator

        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def generator(self):
        return self._generator

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)

    def _compute_losses(self, X, noise):
        generated_images = self._generator(noise, training=True)
        real_output = self._discriminator(X, training=True)
        fake_output = self._discriminator(generated_images, training=True)
        gen_loss = self._generator_loss(fake_output)
        disc_loss = self._discriminator_loss(real_output, fake_output)
        return disc_loss, gen_loss

    @tf.function
    def training_step(self, X):
        batch_size = X.shape[0]
        noise_shape = (batch_size,) + self._generator.input_shape[1:]
        noise = tf.random.normal(noise_shape)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self._compute_losses(X, noise)

        generator_gradients = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)

        self._generator_optimizer.apply_gradients(zip(generator_gradients, self._generator.trainable_variables))
        self._discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self._discriminator.trainable_variables))
        return gen_loss, disc_loss

    def train(self, X, epochs=10, batch_size=128, noise=None, generate_frequency=1,
              save_frequency=None, save_dir=None):
        n_batches = np.ceil(X.shape[0] / batch_size).astype(np.int)
        generated = []

        gen_losses, disc_losses = [], []

        for epoch in range(epochs):
            logger.info("Epoch %d", epoch)
            gen_losses.append(0), disc_losses.append(0)
            for batch in range(n_batches):
                gen_loss, disc_loss = self.training_step(X[batch * batch_size: (batch + 1) * batch_size])
                gen_losses[-1] += gen_loss
                disc_losses[-1] += disc_loss
            gen_losses[-1] /= n_batches
            disc_losses[-1] /= n_batches

            logger.info("Generator loss %f", gen_losses[-1])
            logger.info("Discriminator loss %f", disc_losses[-1])

            if noise is not None and epoch % generate_frequency == 0:
                generated.append(self.generate(noise))

            if save_frequency is not None and epoch % save_frequency == save_frequency - 1:
                save_dir_epoch = os.path.join(save_dir, str(epoch))
                os.makedirs(save_dir_epoch, exist_ok=True)
                logger.info("Saving gan to %s", save_dir_epoch)
                self.save(save_dir_epoch)

        return np.array(generated), np.array(gen_losses), np.array(disc_losses)

    def generate(self, noise):
        return self._generator(noise)

    def save(self, output_directory):
        if not os.path.isdir(output_directory):
            raise ValueError(f"{output_directory} is not directory")
        generator_path = os.path.join(output_directory, "generator.h5")
        discriminator_path = os.path.join(output_directory, "discriminator.h5")

        self._generator.save(generator_path)
        self._discriminator.save(discriminator_path)

    def load(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"{directory} is not directory")
        generator_path = os.path.join(directory, "generator.h5")
        discriminator_path = os.path.join(directory, "discriminator.h5")

        self._generator = tf.keras.load_model(generator_path)
        self._discriminator = tf.keras.load_model(discriminator_path)



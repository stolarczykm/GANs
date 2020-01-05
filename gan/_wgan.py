import tensorflow as tf

from gan import GAN


class WGAN(GAN):
    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        gradient_penalty_strength: float=1.0
    ):
        super().__init__(generator, discriminator)
        self._gradient_penalty_strength = gradient_penalty_strength

    def _compute_losses(self, X, noise):
        generated_images = self._generator(noise, training=True)
        real_output = self._discriminator(X, training=True)
        fake_output = self._discriminator(generated_images, training=True)

        gen_loss = self._generator_loss(fake_output)
        disc_loss = self._discriminator_loss(real_output, fake_output)
        gradient_penalty = self._gradient_penalty(X, generated_images)

        disc_loss = disc_loss + gradient_penalty
        return disc_loss, gen_loss

    def _gradient_penalty(self, X, generated_images):
        epsilon_shape = (X.shape[0],) + (len(X.shape) - 1) * (1,)
        epsilon = tf.random.uniform(epsilon_shape)
        x_hat = epsilon * X + (1 - epsilon) * generated_images
        d_hat = self.discriminator(x_hat)
        gradient = tf.gradients(d_hat, x_hat)
        gradient_penalty = self._gradient_penalty_strength * (tf.norm(gradient) - 1.0) ** 2
        return tf.math.reduce_mean(gradient_penalty)

    def _discriminator_loss(self, real_output, fake_output):
        total_loss = -real_output + fake_output
        return tf.math.reduce_mean(total_loss)

    def _generator_loss(self, fake_output):
        return -tf.math.reduce_mean(fake_output)

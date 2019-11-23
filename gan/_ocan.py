import tensorflow as tf

from gan import GAN


class OCAN(GAN):
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, pretrained_gan: GAN,
                 feature_matching_layer: int):
        assert all(
            layer1.output_shape == layer2.output_shape
            for layer1, layer2 in zip(discriminator.layers, pretrained_gan._discriminator.layers)
        )
        super().__init__(generator, discriminator)
        self._pretrained_discriminator = self._recompile_pretrained_gan(pretrained_gan, feature_matching_layer)

    def _recompile_pretrained_gan(self, pretrained_gan: GAN, feature_matching_layer: int):
        discriminator = pretrained_gan._discriminator
        assert 0 <= feature_matching_layer < len(discriminator.layers)
        assert (
            discriminator.layers[feature_matching_layer].output_shape
            == self._discriminator.layers[feature_matching_layer].output_shape
        )

        return tf.keras.Model(
            inputs=discriminator.input,
            outputs=[discriminator.output, discriminator.layers[feature_matching_layer].output]
        )

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        real_probs = tf.sigmoid(real_output)
        second_real_loss = -tf.reduce_sum(
            tf.multiply(real_probs, tf.math.log(real_probs))
        )
        total_loss = real_loss + fake_loss + 1.85 * second_real_loss
        return total_loss

    @staticmethod
    def pull_away_loss(hidden_state):
        normalized_hidden_state = tf.norm(hidden_state, axis=1)
        normalized_hidden_state_mat = tf.tile(
            tf.expand_dims(normalized_hidden_state, axis=1),
            [1, tf.shape(hidden_state)[1]]
        )
        X = tf.divide(hidden_state, normalized_hidden_state_mat)
        X_X = tf.square(tf.matmul(X, tf.transpose(X)))
        mask = tf.subtract(
            tf.ones_like(X_X),
            tf.linalg.diag(tf.ones([tf.shape(X_X)[0]]))
        )
        pt_loss = tf.divide(
            tf.reduce_sum(tf.multiply(X_X, mask)),
            tf.multiply(
                tf.cast(tf.shape(X_X)[0], tf.float32),
                tf.cast(tf.shape(X_X)[0] - 1, tf.float32)
            )
        )
        return pt_loss

    @staticmethod
    def ent_loss(pretrained_output):
        pretrained_probs = tf.math.sigmoid(pretrained_output)
        threshold = tf.divide(tf.reduce_max(pretrained_probs[:, -1]) +
                              tf.reduce_min(pretrained_probs[:, -1]), 2)
        mask = 0.5 * (tf.sign(tf.subtract(pretrained_probs[:, -1], threshold)) + 1.0)
        loss = tf.reduce_mean(tf.multiply(tf.math.log(pretrained_probs[:, -1]), mask))
        return loss

    @staticmethod
    def feature_matching_loss(real_output, fake_output):
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(
                    tf.square(real_output - fake_output), 1
                )
            )
        )

    def _generator_losses(self, fake_output, real_output, pretrained_output, pretrained_hidden_state):
        # TODO: compute pull_away_loss on hidden state of current discriminator, not pretrained
        pull_away_loss = self.pull_away_loss(pretrained_hidden_state)

        ent_loss = self.ent_loss(pretrained_output)

        # TODO: compute feature_matching loss on some intermediate layer of discriminator
        feature_matching_loss = self.feature_matching_loss(real_output, fake_output)

        return pull_away_loss + ent_loss + feature_matching_loss

    def _compute_losses(self, X, noise):
        generated_images = self._generator(noise, training=True)

        real_output = self._discriminator(X, training=True)
        fake_output = self._discriminator(generated_images, training=True)

        pretrained_output, pretrained_hidden_state = self._pretrained_discriminator(generated_images)

        gen_loss = self._generator_losses(fake_output, real_output, pretrained_output, pretrained_hidden_state)
        disc_loss = self._discriminator_loss(real_output, fake_output)
        return disc_loss, gen_loss
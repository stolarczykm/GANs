import tensorflow as tf
import numpy as np
import functools as fn
from abc import ABC, abstractstaticmethod
from typing import List


def leaky_relu(alpha=0.2, name=None):
    return fn.partial(tf.nn.leaky_relu, alpha=alpha, name=name)


class GAN(ABC):
    def __init__(self, session: tf.Session,
                 batch_size=32, learning_rate=0.001, name=None):
        self.session = session
        self.name = name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._create_networks()
        self._create_losses()
        self._create_optimizers()
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           self.name)
        self.init_vars = tf.initialize_variables(self.variables)

    def train(self, x, epochs):
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            for batch in range(x.shape[0] // self.batch_size):
                x_batch = x[batch * self.batch_size:
                            (batch + 1) * self.batch_size]
                z = np.random.normal(0, 1, (x_batch.shape[0], 100))

                feed_dict = {
                    self.real_image: x_batch,
                    self.noise: z,
                    self.is_train: 1
                }
                for i in range(3):
                    d_loss, _ = self.session.run(
                        (self.d_loss, self.discriminator_opt),
                        feed_dict)
                g_loss, _ = self.session.run((self.g_loss, self.generator_opt),
                                             feed_dict)

                d_losses.append(d_loss), g_losses.append(g_loss)

            print("Epoch {}: gen_loss: {}, disc_loss: {}".format(
                epoch, np.mean(g_losses), np.mean(d_losses)))

    def generate(self, z):
        feed_dict = {
            self.is_train: 0,
            self.noise: z
        }
        return self.session.run(self.generated_image, feed_dict)

    def _create_networks(self):
        self.real_image = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.noise = tf.placeholder(tf.float32, shape=(None, 100))
        self.is_train = tf.placeholder(dtype=tf.bool)

        self.generated_image = self._generator(self.noise, self.is_train)
        self.real_scores = self._discriminator(self.real_image, self.is_train)
        self.fake_scores = self._discriminator(self.generated_image,
                                               self.is_train,
                                               reuse=True)

    def _create_losses(self):
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.real_scores,
                labels=tf.ones([self.batch_size, 1])
            )
        )
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_scores,
                labels=tf.zeros([self.batch_size, 1])
            )
        )
        self.d_loss = 0.5 * (self.d_loss_real + self.d_loss_fake)
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_scores,
                labels=tf.ones([self.batch_size, 1])
            )
        )

    def _create_optimizers(self):
        self.trainable_vars = tf.trainable_variables()  # type: List[tf.Variable]
        self.discriminator_vars = [
            var for var in self.trainable_vars
            if var.name.startswith("discriminator")
        ]
        self.generator_vars = [
            var for var in self.trainable_vars
            if var.name.startswith("generator")
        ]
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.discriminator_opt = tf.train.AdamOptimizer(self.learning_rate,
                                                            beta1=0.5) \
                .minimize(self.d_loss, var_list=self.discriminator_vars)
            self.generator_opt = tf.train.AdamOptimizer(self.learning_rate,
                                                        beta1=0.5) \
                .minimize(self.g_loss, var_list=self.generator_vars)

    @abstractstaticmethod
    def _generator(x, training=True, reuse=False):
        pass

    @abstractstaticmethod
    def _discriminator(x, training=True, reuse=False):
        pass


class WGANGP(GAN, ABC):
    def __init__(self, session: tf.Session,
                 batch_size=32, learning_rate=0.001, name=None,
                 gradient_penalty=10.0):
        self.gradient_penalty_scale = gradient_penalty
        super().__init__(session, batch_size, learning_rate, name)

    def _create_networks(self):
        super()._create_networks()

        epsilon = tf.random_uniform([], 0, 1)
        x_hat = epsilon * self.real_image + (1 - epsilon) * self.generated_image
        d_hat = self._discriminator(x_hat, reuse=True, training=self.is_train)

        self.d_grad = tf.gradients(d_hat, x_hat)[0]

    def _create_losses(self):
        d_grad_norm = tf.sqrt(
            tf.reduce_sum(tf.square(self.d_grad), axis=1))
        d_grad_penalty = tf.reduce_mean(tf.square(d_grad_norm - 1)
                                        * self.gradient_penalty_scale)

        self.d_loss = (tf.reduce_mean(self.fake_scores) -
                       tf.reduce_mean(self.real_scores) +
                       d_grad_penalty)
        self.g_loss = -tf.reduce_mean(self.fake_scores)


class MnistGan(GAN):
    def __init__(self, session: tf.Session,
                 batch_size=32, learning_rate=0.001, name=None):
        super().__init__(session, batch_size, learning_rate, name)

    @staticmethod
    def _generator(x, training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(x, 1024, tf.nn.relu, name="dense1")
            x = tf.layers.batch_normalization(x, training=training, name="bn1")
            x = tf.layers.dense(x, 7 * 7 * 128, tf.nn.relu, name="dense2")
            x = tf.layers.batch_normalization(x, training=training, name="bn2")
            x = tf.reshape(x, (-1, 7, 7, 128), name="reshape1")
            x = tf.layers.conv2d_transpose(x, 64, (4, 4), (2, 2),
                                           padding="same",
                                           activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x, training=training, name="bn3")
            x = tf.layers.conv2d_transpose(x, 1, (4, 4), (2, 2),
                                           padding="same",
                                           activation=tf.nn.tanh)
            return x

    @staticmethod
    def _discriminator(x, training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.layers.conv2d(x, 64, (4, 4), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv1")
            x = tf.layers.conv2d(x, 128, (4, 4), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv2")
            x = tf.layers.batch_normalization(x, training=training, name="bn1")
            x = tf.reshape(x, shape=(-1, 7 * 7 * 128), name="reshape1")
            x = tf.layers.dense(x, 1024, activation=leaky_relu(0.1))
            x = tf.layers.batch_normalization(x, training=training, name="bn2")
            x = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)
            return x


class MnistWgan(WGANGP):
    def __init__(self, session: tf.Session,
                 batch_size=32, learning_rate=0.001, name=None,
                 gradient_penalty=10.0):
        super().__init__(session, batch_size, learning_rate, name,
                         gradient_penalty=gradient_penalty)

    @staticmethod
    def _generator(x, training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.layers.dense(x, 1024, tf.nn.relu, name="dense1")
            x = tf.layers.batch_normalization(x, training=training, name="bn1")
            x = tf.layers.dense(x, 7 * 7 * 128, tf.nn.relu, name="dense2")
            x = tf.layers.batch_normalization(x, training=training, name="bn2")
            x = tf.reshape(x, (-1, 7, 7, 128), name="reshape1")
            x = tf.layers.conv2d_transpose(x, 64, (4, 4), (2, 2),
                                           padding="same",
                                           activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x, training=training, name="bn3")
            x = tf.layers.conv2d_transpose(x, 1, (4, 4), (2, 2),
                                           padding="same",
                                           activation=tf.nn.tanh)
            return x

    @staticmethod
    def _discriminator(x, training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.layers.conv2d(x, 64, (4, 4), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv1")
            x = tf.layers.conv2d(x, 128, (4, 4), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv2")
            # x = tf.layers.batch_normalization(x, training=training, name="bn1")
            x = tf.reshape(x, shape=(-1, 7 * 7 * 128), name="reshape1")
            x = tf.layers.dense(x, 1024, activation=leaky_relu(0.1))
            # x = tf.layers.batch_normalization(x, training=training, name="bn2")
            x = tf.layers.dense(x, 1)
            return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 128.0 - 1, x_test / 255.0 - 1
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    sess = tf.Session()
    gan = MnistGan(sess, learning_rate=0.001)
    sess.run(gan.init_vars)
    noise = np.random.normal(0, 1, (1, 100))
    for i in range(10):
        gan.train(x_train[:1000], 10)
        pr = gan.generate(noise)
        plt.imshow(pr[0, :, :, 0], cmap="gray_r")
        plt.show()

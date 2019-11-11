import abc
import functools as fn
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from typing import List


def leaky_relu(alpha=0.2, name=None):
    return fn.partial(tf.nn.leaky_relu, alpha=alpha, name=name)


class GAN(abc.ABC):
    def __init__(self, session: tf.Session, learning_rate=0.001, name=None):
        self.session = session
        self.name = name
        self.learning_rate = learning_rate
        with tf.variable_scope(self.name):
            self._create_networks()
            self._create_losses()
            self._create_optimizers()

        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           self.name)
        self.init_vars = tf.variables_initializer(self.variables)
        self.summaries = tf.summary.merge_all()
        self.images_placeholder = tf.placeholder(tf.float32,
                                                 shape=(None, 112, 140, 1))
        self.image_summary = tf.summary.image('generated_images',
                                              self.images_placeholder)

        self.session.run([self.init_vars])

    def train(self, x, epochs, batch_size, file_writer=None):
        """

        :type x: np.ndarray
        :type epochs: int
        :type file_writer: tf.summary.FileWriter
        :type batch_size: int
        :return:
        """
        for epoch in range(epochs):
            for batch in range(x.shape[0] // batch_size):
                x_batch = x[batch * batch_size: (batch + 1) * batch_size]
                z = np.random.normal(0, 1, (x_batch.shape[0], 100))

                feed_dict = {
                    self.real_image: x_batch,
                    self.noise: z,
                    self.is_train: 1
                }
                for i in range(3):
                    d_loss, _, summaries = self.session.run(
                        (self.d_loss, self.discriminator_opt, self.summaries),
                        feed_dict)
                    if file_writer is not None:
                        file_writer.add_summary(summaries, epoch)

                g_loss, _, summaries = self.session.run(
                    (self.g_loss, self.generator_opt, self.summaries),
                    feed_dict)
                if file_writer is not None:
                    file_writer.add_summary(summaries, epoch)

    def generate(self, z):
        feed_dict = {
            self.is_train: 0,
            self.noise: z
        }
        return self.session.run(self.generated_image, feed_dict)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

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
                labels=tf.ones_like(self.real_scores)
            )
        )
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_scores,
                labels=tf.zeros_like(self.fake_scores)
            )
        )
        self.d_loss = 0.5 * (self.d_loss_real + self.d_loss_fake)
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_scores,
                labels=tf.ones_like(self.fake_scores)
            )
        )

    def _create_optimizers(self):
        self.trainable_vars = tf.trainable_variables()  # type: List[tf.Variable]
        self.discriminator_vars = [
            var for var in self.trainable_vars
            if var.name.startswith("{}/discriminator".format(self.name))
        ]
        self.generator_vars = [
            var for var in self.trainable_vars
            if var.name.startswith("{}/generator".format(self.name))
        ]
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

            d_grads = opt.compute_gradients(self.d_loss,
                                            var_list=self.discriminator_vars)
            self.discriminator_opt = opt.apply_gradients(d_grads)
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
            g_grads = opt.compute_gradients(self.g_loss,
                                            var_list=self.generator_vars)
            self.generator_opt = opt.apply_gradients(g_grads)

    @staticmethod
    @abc.abstractmethod
    def _generator(x, training=True, reuse=False):
        pass

    @staticmethod
    @abc.abstractmethod
    def _discriminator(x, training=True, reuse=False):
        pass


class WGANGP(GAN, abc.ABC):
    def __init__(self, session: tf.Session,
                 learning_rate=0.001, name=None,
                 gradient_penalty=10.0):
        self.gradient_penalty_scale = gradient_penalty
        super().__init__(session, learning_rate, name)

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
                 learning_rate=0.001, name=None):
        super().__init__(session, learning_rate, name)

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


class CWGANGP(WGANGP):
    def _create_losses(self):
        epsilon = tf.random_uniform([], 0, 1)
        x_hat = epsilon * self.real_image + (1 - epsilon) * self.generated_image
        d_hat = self._discriminator([x_hat, self.condition],
                                    reuse=True, training=self.is_train)
        self.d_grad = tf.gradients(d_hat, x_hat)[0]

        d_grad_norm = tf.sqrt(
            tf.reduce_sum(tf.square(self.d_grad), axis=1))
        d_grad_penalty = tf.reduce_mean(tf.square(d_grad_norm - 1)
                                        * self.gradient_penalty_scale)

        self.d_loss = (tf.reduce_mean(self.fake_scores) -
                       tf.reduce_mean(self.real_scores) +
                       d_grad_penalty)
        tf.summary.scalar("discriminator_loss", self.d_loss)
        self.g_loss = -tf.reduce_mean(self.fake_scores)
        tf.summary.scalar("generator_loss", self.g_loss)

    def _create_networks(self):
        self.real_image = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.noise = tf.placeholder(tf.float32, shape=(None, 64))
        self.condition = tf.placeholder(tf.float32, shape=(None, 10))
        self.is_train = tf.placeholder(dtype=tf.bool)

        self.generated_image = self._generator(
            [self.noise, self.condition], self.is_train)
        self.real_scores = self._discriminator(
            [self.real_image, self.condition], self.is_train)
        self.fake_scores = self._discriminator(
            [self.generated_image, self.condition], self.is_train,
            reuse=True)


class MnistWgan(WGANGP):
    def __init__(self, session: tf.Session, learning_rate=0.001, name=None,
                 gradient_penalty=10.0):
        super().__init__(session, learning_rate, name,
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


class MnistCWGAN(CWGANGP):
    def __init__(self, session: tf.Session, learning_rate=0.001, name=None,
                 gradient_penalty=10.0):
        super().__init__(session, learning_rate, name,
                         gradient_penalty=gradient_penalty)

    @staticmethod
    def _generator(x, training=True, reuse=False):
        [x, condition] = x
        with tf.variable_scope("generator", reuse=reuse):
            x = tf.concat([x, condition], axis=1, name="concat1")
            x = tf.layers.dense(x, 4*4*512, tf.identity, name="linear1")
            x = tf.reshape(x, [-1, 4, 4, 512], name="reshape1")
            x = tf.layers.batch_normalization(x, training=training, name="bn1")
            x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2),
                                           padding="same",
                                           activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x, training=training, name="bn2")
            x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2),
                                           padding="same",
                                           activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x, training=training, name="bn3")
            x = tf.layers.conv2d_transpose(x, 1, (5, 5), (2, 2),
                                           padding="same",
                                           activation=tf.nn.tanh)
            x = x[:, 2:-2, 2:-2, :]
            return x

    @staticmethod
    def _discriminator(x, training=True, reuse=False):
        [x, condition] = x
        with tf.variable_scope("discriminator", reuse=reuse):

            condition = tf.reshape(condition, [-1, 1, 1, 10], "cond_reshape")
            condition = tf.tile(condition, [1, 28, 28, 1], "tile")

            x = tf.concat([x, condition], axis=-1, name="concat1")
            x = tf.layers.conv2d(x, 64, (5, 5), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv1")
            x = tf.layers.conv2d(x, 128, (5, 5), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv2")
            x = tf.layers.conv2d(x, 256, (5, 5), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv3")
            x = tf.layers.conv2d(x, 512, (5, 5), (2, 2),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv4")
            x = tf.layers.conv2d(x, 1, (4, 4), (1, 1),
                                 padding="same",
                                 activation=leaky_relu(0.1),
                                 name="conv5")
            x = tf.layers.average_pooling2d(x, (2, 2), (1, 1), name="output")
            x = tf.layers.dense(x, 1, name='output')
            return x

    def train(self, x, epochs, batch_size, file_writer=None):
        x, cond = x
        conditions = to_categorical(
            np.concatenate([np.arange(10), np.arange(10)]))
        noise = np.random.normal(0, 1, (20, 64))

        bs = batch_size
        for epoch in range(epochs):
            print(epoch, end="\r")
            for batch in range(x.shape[0] // bs):
                ind = slice(batch * bs, (batch+1) * bs)
                x_batch = x[ind]
                cond_batch = cond[ind]
                z = np.random.normal(0, 1, (x_batch.shape[0], 64))
                feed_dict = {
                    self.real_image: x_batch,
                    self.noise: z,
                    self.is_train: 1,
                    self.condition: cond_batch
                }
                for i in range(3):
                    d_loss, _, summaries = self.session.run(
                        (self.d_loss, self.discriminator_opt, self.summaries),
                        feed_dict)
                    if file_writer is not None:
                        file_writer.add_summary(summaries, epoch)

                g_loss, _, summaries = self.session.run(
                    (self.g_loss, self.generator_opt, self.summaries),
                    feed_dict)
                if file_writer is not None:
                    file_writer.add_summary(summaries, epoch)
            images = self.generate([noise, conditions]).reshape(4, 5, 28, 28)
            images = np.concatenate(np.concatenate(images, axis=1), axis=1)
            images = images.reshape(-1, 4*28, 5*28, 1)
            summary, = self.session.run([self.image_summary],
                                        {self.images_placeholder: images})
            file_writer.add_summary(summary, epoch)

    def generate(self, z):
        z, cond = z
        feed_dict = {self.is_train: 0, self.noise: z, self.condition: cond}
        return self.session.run(self.generated_image, feed_dict)


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

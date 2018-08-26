import numpy as np
from keras.layers import Dense, Conv2D, Deconv2D, Flatten, MaxPool2D, Reshape, Concatenate
from keras.models import Input, Model
from keras.optimizers import Adam


class GAN:
    def __init__(self,
                 input_dim=2,
                 noise_dim=2,
                 discriminator_units=(10, 10),
                 generator_units=(10, 10)):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.discriminator = self._create_discriminator(discriminator_units)
        self.generator = self._create_generator(generator_units)
        self.optimizer = Adam(1e-4, decay=1e-6)
        self._compile_models()
        self.generator.summary()
        self.discriminator.summary()
        self.combined_model = self._create_combined_model()
        self._compile_combined_model()

    def train(self, X, epochs, batch_size):
        half_batch_size = int(batch_size / 2)
        batches_per_epoch = 2 * int(X.shape[0] / batch_size)
        for i in range(epochs):
            permutation = np.random.choice(np.arange(X.shape[0]), X.shape[0],
                                           replace=False)
            X = X[permutation]
            generator_losses, discriminator_losses = [], []
            generator_accs, discriminator_accs = [], []
            for j in range(batches_per_epoch):
                noise = np.random.normal(0, 1, (half_batch_size, self.noise_dim))
                x_fake_batch = self.generator.predict(noise)
                x_real_batch = X[j*half_batch_size: (j+1)*half_batch_size]
                x_batch = np.concatenate([x_fake_batch, x_real_batch],
                                         axis=0)
                y_batch = np.concatenate([np.zeros(half_batch_size),
                                          np.ones(half_batch_size)])
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(x_batch, y_batch)
                self.discriminator.trainable = False
                noise = np.random.normal(0, 1, (half_batch_size, self.noise_dim))
                g_loss = self.combined_model.train_on_batch(
                    noise,
                    np.ones(half_batch_size))
                generator_losses.append(g_loss[0])
                discriminator_losses.append(d_loss[0])
                generator_accs.append(g_loss[1])
                discriminator_accs.append(d_loss[1])
            print(np.mean(generator_losses), ",", np.mean(discriminator_losses),
                  end=", ")
            print(np.mean(generator_accs), ",", np.mean(discriminator_accs))

    def _create_discriminator(self, hidden_units):
        inputs = Input(shape=self.input_dim)
        # hidden_state = Dense(hidden_units[0], activation="relu")(inputs)
        # for n_units in hidden_units[1:]:
        #     hidden_state = Dense(n_units, activation="relu")(hidden_state)
        # outputs = Dense(1, activation="sigmoid")(hidden_state)
        hidden_state = Conv2D(hidden_units[0], (3,3), padding="same", activation="relu")(inputs)
        for n_units in hidden_units[1:]:
            hidden_state = Conv2D(n_units, (3, 3), padding="same", activation="relu")(hidden_state)
            hidden_state = MaxPool2D(2)(hidden_state)
        hidden_state = Flatten()(hidden_state)
        hidden_state = Dense(20)(hidden_state)
        outputs = Dense(1, activation="sigmoid")(hidden_state)
        return Model(inputs, outputs)

    def _create_generator(self, hidden_units):
        inputs = Input(shape=(self.noise_dim,))
        hidden_state = Dense(hidden_units[0], activation="relu")(inputs)
        mult = 2**(len(hidden_units) - 1)
        hidden_state = Reshape((28//mult, 28//mult, 3))(hidden_state)
        for n_units in hidden_units[1:]:
            print(hidden_state)
            hidden_state = Deconv2D(
                n_units, (3, 3), strides=2, padding="same", activation="relu"
            )(hidden_state)
            hidden_state = Conv2D(
                n_units, (3, 3), padding="same", activation="relu"
            )(hidden_state)
        outputs = Conv2D(1, (3, 3), padding="same")(hidden_state)
        return Model(inputs, outputs)

    def _create_combined_model(self):
        noise = Input(shape=(self.noise_dim,))
        fake_img = self.generator(noise)
        outputs = self.discriminator(fake_img)
        return Model(noise, outputs)

    def _compile_models(self):
        for model in [self.discriminator, self.generator]:
            model.compile(loss="binary_crossentropy",
                          optimizer=self.optimizer,
                          metrics=["accuracy"])

    def _compile_combined_model(self):
        self.discriminator.trainable = False
        self.combined_model.compile(loss="binary_crossentropy",
                                    optimizer=self.optimizer,
                                    metrics=["accuracy"])


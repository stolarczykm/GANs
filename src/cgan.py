import numpy as np
from keras.models import Input, Model
from keras.optimizers import Adam


class CGAN:
    def __init__(self, create_generator_func, create_discriminator_func,
                 input_dim=2, noise_dim=2, condition_dim=1):

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.noise_dim = noise_dim
        self.discriminator = create_discriminator_func(input_dim, condition_dim)
        self._compile_models()
        self.generator = create_generator_func(noise_dim, condition_dim)
        self.discriminator.trainable = False
        self.combined_model = self._create_combined_model()
        self._compile_combined_model()
        self.generator.summary()
        self.discriminator.summary()

    def train(self, X, C, epochs, batch_size):
        half_batch_size = int(batch_size / 2)
        batches_per_epoch = 2 * int(X.shape[0] / batch_size)
        for i in range(epochs):
            permutation = np.random.choice(np.arange(X.shape[0]), X.shape[0],
                                           replace=False)
            X = X[permutation]
            C = C[permutation]
            generator_losses, discriminator_losses = [], []
            generator_accs, discriminator_accs = [], []
            for j in range(batches_per_epoch):
                noise = np.random.normal(0, 1, (half_batch_size, self.noise_dim))
                C_batch = C[j*half_batch_size: (j+1)*half_batch_size]
                x_fake_batch = self.generator.predict([C_batch, noise])
                x_real_batch = X[j*half_batch_size: (j+1)*half_batch_size]
                d_loss = self.discriminator.train_on_batch(
                    [np.concatenate([C_batch, C_batch], axis=0),
                     np.concatenate([x_real_batch, x_fake_batch], axis=0)],
                    np.concatenate([np.ones(half_batch_size),
                                    np.zeros(half_batch_size)],
                                   axis=0)
                )
                new_noise = np.random.normal(0, 1, (half_batch_size, self.noise_dim))

                g_loss = self.combined_model.train_on_batch(
                    [np.concatenate([C_batch, C_batch]),
                     np.concatenate([noise, new_noise])],
                    0.95 * np.ones(batch_size))
                generator_losses.append(g_loss[0])
                discriminator_losses.append(d_loss[0])
                generator_accs.append(g_loss[1])
                discriminator_accs.append(d_loss[1])
            print(np.mean(generator_losses), ",", np.mean(discriminator_losses),
                  end=", ", sep='')
            print(np.mean(generator_accs), ",", np.mean(discriminator_accs),
                  sep="")

    def _create_combined_model(self):
        noise = Input(shape=(self.noise_dim,))
        condition = Input(shape=(self.condition_dim,))
        fake_img = self.generator([condition, noise])
        outputs = self.discriminator([condition, fake_img])
        return Model([condition, noise], outputs)

    def _compile_models(self):
        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=Adam(5e-4, clipvalue=1, beta_1=.5),
                                   metrics=["accuracy"])

    def _compile_combined_model(self):
        self.combined_model.compile(loss="binary_crossentropy",
                                    optimizer=Adam(5e-4, clipvalue=1, beta_1=.5),
                                    metrics=["accuracy"])


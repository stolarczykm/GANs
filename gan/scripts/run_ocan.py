import os

import numpy as np

from gan import GAN, OCAN


def run_ocan():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    from gan import get_data_directory

    data_benign = np.load(os.path.join(get_data_directory(), "credit_card", "ben_hid_repre_r2.npy"))
    data_vandal = np.load(os.path.join(get_data_directory(), "credit_card", "van_hid_repre_r2.npy"))

    data_benign_train = data_benign[:1000]
    data_benign_test = data_benign[-490:]

    print(data_benign.shape)
    print(data_vandal.shape)

    noise_dim = 50
    data_dim = data_benign_train.shape[1]

    regular_gan = GAN(make_generator(noise_dim, data_dim), make_discriminator(data_dim))
    _, gen_loss, disc_loss = regular_gan.train(data_benign_train, 2000, 512)
    plt.figure()
    plt.plot(gen_loss)
    plt.plot(disc_loss)
    plt.title("GAN losses")

    print("Training OCAN")
    ocan = OCAN(make_generator(noise_dim, data_dim), make_discriminator(data_dim), regular_gan, 1)
    _, ocan_gen_loss, ocan_disc_loss = ocan.train(data_benign_train, epochs=200, batch_size=70)

    plt.figure()
    plt.plot(ocan_gen_loss)
    plt.plot(ocan_disc_loss)
    plt.title("OCAN losses")

    test_predictions_benign = ocan.discriminator.predict(data_benign_test)
    test_predictions_vandal = ocan.discriminator.predict(data_vandal)

    predictions = np.concatenate([test_predictions_benign, test_predictions_vandal])
    labels = np.concatenate([np.ones_like(test_predictions_benign), np.zeros_like(test_predictions_vandal)])
    roc = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    plt.figure()
    plt.plot(*roc[:2])
    plt.title(f"AUC: {100*auc:.2f}")
    plt.show()


def make_discriminator(input_length):
    from tensorflow.keras import models, layers

    return models.Sequential([
        layers.Dense(100, activation="relu", input_shape=(input_length,)),
        layers.Dense(50, activation="relu"),
        layers.Dense(1)
    ])


def make_generator(noise_length, output_length):
    from tensorflow.keras import models, layers

    return models.Sequential([
        layers.Dense(50, activation="relu", input_shape=(noise_length,)),
        layers.Dense(100, activation="relu"),
        layers.Dense(output_length)
    ])


if __name__ == '__main__':
    run_ocan()

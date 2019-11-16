import datetime
import os

import numpy as np
from sklearn.datasets import make_moons

from gan import GAN, OCAN, get_models_directory
from gan.animation import ScatterAnimation


def run_ocan_on_credit_card():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    from gan import get_data_directory

    # data_benign = np.load(os.path.join(get_data_directory(), "credit_card", "ben_hid_repre_r2.npy"))
    # data_vandal = np.load(os.path.join(get_data_directory(), "credit_card", "van_hid_repre_r2.npy"))

    data_benign = np.load(os.path.join(get_data_directory(), "credit_card_raw", "ben_raw_r0.npy"))
    data_vandal = np.load(os.path.join(get_data_directory(), "credit_card_raw", "van_raw_r0.npy"))

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


def run_ocan_on_moons():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_dir = os.path.join(get_models_directory(), run_name)
    os.makedirs(model_dir, exist_ok=False)

    data_train, y_train = make_moons(2000, noise=0.05)
    data_test, y_test = make_moons(800, noise=0.05)
    data_benign_train = data_train[y_train == 0]

    noise_dim = 16
    data_dim = data_benign_train.shape[1]
    noise = np.random.normal(size=(400, noise_dim))

    regular_gan = GAN(
        make_generator(noise_dim, data_dim, (20, 20, 20)),
        make_discriminator(data_dim, (20, 20, 20))
    )
    generated, gen_loss, disc_loss = regular_gan.train(data_benign_train, 1, 128, generate_frequency=10, noise=noise)
    # plt.figure()
    # plt.plot(gen_loss)
    # plt.plot(disc_loss)
    # plt.title("GAN losses")
    # plt.savefig(os.path.join(model_dir, "gan_losses.png"))

    print("Training OCAN")
    ocan = OCAN(
        make_generator(noise_dim, data_dim, (20, 20, 20)),
        make_discriminator(data_dim, (20, 20, 20)),
        regular_gan,
        2)
    ocan_generated, ocan_gen_loss, ocan_disc_loss = ocan.train(data_benign_train, epochs=1000, batch_size=16,
                                                               generate_frequency=10, noise=noise)

    plt.figure()
    plt.plot(ocan_gen_loss)
    plt.plot(ocan_disc_loss)
    plt.title("OCAN losses")
    plt.savefig(os.path.join(model_dir, "OCAN_losses.png"))

    predictions = ocan.discriminator.predict(data_test)
    roc = roc_curve(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    plt.figure()
    plt.plot(*roc[:2])
    plt.title(f"AUC: {100*auc:.2f}")
    plt.savefig(os.path.join(model_dir, "auc.png"))

    # animation = ScatterAnimation(5000)
    # animation.animate(data_benign_train, generated, gen_loss, disc_loss)
    # animation.save(os.path.join(model_dir, "gan_animation.mp4"))

    animation = ScatterAnimation(5000)
    animation.animate(data_train, ocan_generated, ocan_gen_loss, ocan_disc_loss, real_data_colors=y_train)
    animation.save(os.path.join(model_dir, "ocan_animation.mp4"))


def make_discriminator(input_length, layer_sizes=(100, 50)):
    from tensorflow.keras import models, layers

    model = models.Sequential([])
    for i, n_units in enumerate(layer_sizes):
        kwargs = {"input_shape": (input_length, )} if i == 0 else {}
        layer = layers.Dense(n_units, activation="relu", **kwargs)
        model.add(layer)
    model.add(layers.Dense(1))
    return model


def make_generator(noise_length, output_length, layer_sizes=(50, 100)):
    from tensorflow.keras import models, layers

    model = models.Sequential([])
    for i, n_units in enumerate(layer_sizes):
        kwargs = {"input_shape": (noise_length, )} if i == 0 else {}
        layer = layers.Dense(n_units, activation="relu", **kwargs)
        model.add(layer)
    model.add(layers.Dense(output_length))
    return model


if __name__ == '__main__':
    run_ocan_on_credit_card()

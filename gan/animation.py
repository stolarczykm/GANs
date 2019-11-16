import abc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Animation(abc.ABC):
    def __init__(self, animation_length: int = 10000):
        """
        :param animation_length: Animation length in milliseconds

        """
        self._animation = None
        self._figure = None
        self._animation_length = animation_length

    @abc.abstractmethod
    def animate(self, real_data, generated_data, generator_loss, discriminator_loss, **kwargs):
        pass

    def save(self, path):
        self._animation.save(path)
        plt.close(self._figure)


class ScatterAnimation(Animation):
    def animate(self, real_data, generated_data, generator_loss, discriminator_loss, **kwargs):
        assert len(real_data.shape) == 2
        assert real_data.shape[1] == 2

        assert len(generated_data.shape) == 3
        assert generated_data.shape[2] == 2

        real_data_colors = kwargs.get("real_data_colors")

        ax, ax2, ax3, self._figure = self._create_axes()
        self._set_fig_size(self._figure)
        sc, txt = self._plot_real_data(ax, real_data, real_data_colors)
        line_gen = self._plot_generator_loss(ax2, generator_loss)
        line_disc = self._plot_discriminator_loss(ax2, ax3, discriminator_loss)

        def animate(i):
            sc.set_offsets(generated_data[i])
            epoch = int(i * len(generator_loss) / len(generated_data))
            txt.set_text(str(epoch))
            line_gen.set_data(np.arange(0, epoch), generator_loss[:epoch])
            line_disc.set_data(np.arange(0, epoch), discriminator_loss[:epoch])
            return sc, txt, line_gen, line_disc

        self._animation = animation.FuncAnimation(
            self._figure,
            animate,
            frames=len(generated_data),
            interval=self._animation_length // len(generated_data),
            blit=True
        )

    @staticmethod
    def _set_fig_size(fig):
        fig.set_figheight(10)
        fig.set_figwidth(10)

    @staticmethod
    def _plot_discriminator_loss(ax2, ax3, discriminator_loss):
        line_disc, = ax3.plot([], color="orange")
        ax3.set_xlim(0, len(discriminator_loss))
        ax3.set_ylim(min(discriminator_loss), max(discriminator_loss))
        ax2.set_ylabel("discriminator loss")
        return line_disc

    @staticmethod
    def _plot_generator_loss(ax2, generator_loss):
        line_gen, = ax2.plot([], label="generator")
        ax2.plot([], label="discriminator")
        ax2.set_xlim(0, len(generator_loss))
        ax2.set_ylim(min(generator_loss), max(generator_loss))
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("generator loss")
        ax2.legend()
        return line_gen

    @staticmethod
    def _plot_real_data(ax, real_data, real_data_colors=None):
        ax.axis('off')
        if real_data_colors is not None:
            ax.scatter(real_data[:, 0], real_data[:, 1], label="real", c=real_data_colors)
        else:
            ax.scatter(real_data[:, 0], real_data[:, 1], label="real")
        ax.set_xlim([-1.4, 1.4])
        ax.set_ylim([-1.4, 1.4])
        txt = ax.text(-1.2, 1.2, "")
        sc = ax.scatter([], [], label="generated")
        return sc, txt

    @staticmethod
    def _create_axes():
        fig, (ax, ax2) = plt.subplots(2, 1, tight_layout=True, gridspec_kw={'height_ratios': [4, 1]})
        ax3 = ax2.twinx()
        return ax, ax2, ax3, fig


class ImageAnimation(Animation):
    def animate(self, real_data, generated_data, generator_loss, discriminator_loss, **kwargs):
        assert len(generated_data.shape) == 5

        ax, ax2, ax3, self._figure = self._create_axes()
        self._set_fig_size(self._figure)
        im = self._show_images(ax, generated_data)
        line_gen = self._plot_generator_loss(ax2, generator_loss)
        line_disc = self._plot_discriminator_loss(ax2, ax3, discriminator_loss)

        def animate(i):
            im.set_data(self._images_to_grid(generated_data[i]))
            epoch = int(i * len(generator_loss) / len(generated_data))
            line_gen.set_data(np.arange(0, epoch), generator_loss[:epoch])
            line_disc.set_data(np.arange(0, epoch), discriminator_loss[:epoch])
            return im, line_gen, line_disc

        self._animation = animation.FuncAnimation(
            self._figure,
            animate,
            frames=len(generated_data),
            interval=self._animation_length // len(generated_data),
            blit=True
        )

    @staticmethod
    def _set_fig_size(fig):
        fig.set_figheight(10)
        fig.set_figwidth(10)

    @staticmethod
    def _plot_discriminator_loss(ax2, ax3, discriminator_loss):
        line_disc, = ax3.plot([], color="orange")
        ax3.set_xlim(0, len(discriminator_loss))
        ax3.set_ylim(min(discriminator_loss), max(discriminator_loss))
        ax2.set_ylabel("discriminator loss")
        return line_disc

    @staticmethod
    def _plot_generator_loss(ax2, generator_loss):
        line_gen, = ax2.plot([], label="generator")
        ax2.plot([], label="discriminator")
        ax2.set_xlim(0, len(generator_loss))
        ax2.set_ylim(min(generator_loss), max(generator_loss))
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("generator loss")
        ax2.legend()
        return line_gen

    @classmethod
    def _show_images(cls, ax, generated_data):
        ax.axis('off')
        im = ax.imshow(cls._images_to_grid(generated_data[0]), cmap="gray")
        return im

    @staticmethod
    def _images_to_grid(images):
        initial_shape = images.shape
        n_images = len(images)
        image_shape = images.shape[1:]
        sqrt_n_images = int(np.ceil(np.sqrt(n_images)))
        if n_images < sqrt_n_images**2:
            images = np.concatenate([images, np.zeros((sqrt_n_images**2 - n_images,) + image_shape)], axis=0)
        images = images.reshape((sqrt_n_images, sqrt_n_images, ) + image_shape)
        images = np.concatenate(images, axis=1)
        images = np.concatenate(images, axis=1)
        assert images.shape[:2] == (sqrt_n_images * initial_shape[1], sqrt_n_images * initial_shape[2])
        return images[..., 0]

    @staticmethod
    def _create_axes():
        fig, (ax, ax2) = plt.subplots(2, 1, tight_layout=True, gridspec_kw={'height_ratios': [4, 1]})
        ax3 = ax2.twinx()
        return ax, ax2, ax3, fig



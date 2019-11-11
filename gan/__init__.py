import os
from gan._gan import GAN


def get_gan_package_root_direcotry():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_models_directory():
    return os.path.join(get_gan_package_root_direcotry(), "models")


def get_animations_directory():
    return os.path.join(get_gan_package_root_direcotry(), "animations")


def get_data_directory():
    return os.path.join(get_gan_package_root_direcotry(), "data")

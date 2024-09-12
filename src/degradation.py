import glob
import os
import random
from enum import Enum, auto

import cv2
import numpy as np
import torch
from torchvision.utils import save_image

from .utils import read_image

BRIGHTNESS_LOW = -1
BRIGHTNESS_HIGH = 1
NOISE_LOW = 0
NOISE_HIGH = 0.6
NOISE_MEAN = 0.15
NOISE_STDDEV = 0.35


class Defect(Enum):
    BRIGHTNESS = auto()
    BLUR = auto()
    NOISE = auto()


def randomly_degrade(
    tensor,
    noise_coef=None,
    distribution=None,
    kernel_size=None,
    blur_type=None,
    brightness=None,
    light_enabled=True,
    blur_enabled=True,
    noise_enabled=True,
):
    min_val, max_val = tensor.min(), tensor.max()
    tensor = (tensor - min_val) / (max_val - min_val)

    defects = []
    if light_enabled:
        defects.append(Defect.BRIGHTNESS)
    if blur_enabled:
        defects.append(Defect.BLUR)
    if noise_enabled:
        defects.append(Defect.NOISE)

    if defects:
        original_defect = random.choice(defects)

    for defect in defects:
        if defect is not original_defect:
            is_present = np.random.rand() < 0.4
        else:
            is_present = True

        if is_present:
            match defect:
                case Defect.BRIGHTNESS:
                    tensor = degrade_brightness(tensor, brightness=brightness)
                case Defect.BLUR:
                    tensor = apply_blur(
                        tensor, blur_type=blur_type, kernel_size=kernel_size
                    )
                case Defect.NOISE:
                    tensor = add_noise(
                        tensor, distribution=distribution, noise_coef=noise_coef
                    )

    return tensor


def degrade_brightness(tensor, brightness=None):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img + brightness / 2.5, 0, 1)
    return torch.tensor(img).permute(2, 0, 1).float()


def apply_blur(tensor, blur_type=None, kernel_size=None):
    img = tensor.permute(1, 2, 0).cpu().numpy()

    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7, 9, 11, 13])

    if blur_type is None:
        blur_type = random.choice(["Gaussian", "Box"])

    if blur_type == "Gaussian":
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif blur_type == "Box":
        img = cv2.blur(img, (kernel_size, kernel_size))

    return torch.tensor(img).permute(2, 0, 1).float()


def add_noise(tensor, distribution=None, noise_coef=0.5):
    img = tensor.permute(1, 2, 0).cpu().numpy()

    if distribution is None:
        distribution = random.choice(["Gaussian", "Uniform"])

    if distribution == "Gaussian":
        noise = np.random.normal(NOISE_MEAN, NOISE_STDDEV, img.shape)
    elif distribution == "Uniform":
        noise = np.random.uniform(NOISE_LOW, NOISE_HIGH, img.shape)

    img = np.clip(img + noise_coef * noise, 0, 1)
    return torch.tensor(img).permute(2, 0, 1).float()


def degrade_all_images(destination, config, leave_intact=0.2):
    """Apply stochastic degradation pipeline to images in given dir leaving some of them intact"""
    # destination has to have particular directory structure (see in main.ipynb)
    train_images = glob.glob(os.path.join(destination, "train_output") + "/*.*")
    test_images = glob.glob(os.path.join(destination, "test_output") + "/*.*")
    degradation_params = dict(
        noise_coef=config.noise_coef,
        distribution=config.distribution,
        kernel_size=config.kernel_size,
        blur_type=config.blur_type,
        brightness=config.brightness,
        light_enabled=config.light_enabled,
        blur_enabled=config.blur_enabled,
        noise_enabled=config.noise_enabled,
    )

    for img_path in train_images:
        dst_path = os.path.join(destination, "train_input", os.path.basename(img_path))
        img = read_image(img_path)
        degraded_img = (
            randomly_degrade(img, **degradation_params)
            if np.random.random() >= leave_intact
            else img
        )
        save_image(degraded_img, dst_path)

    for img_path in test_images:
        dst_path = os.path.join(destination, "test_input", os.path.basename(img_path))
        img = read_image(img_path)
        degraded_img = (
            randomly_degrade(img, **degradation_params)
            if np.random.random() >= leave_intact
            else img
        )
        save_image(degraded_img, dst_path)

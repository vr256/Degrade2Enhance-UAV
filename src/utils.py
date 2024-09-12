import os
import random
import shutil
from dataclasses import dataclass, field
from functools import wraps
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2


def singleton(cls):
    obj = None

    @wraps(cls)
    def wrapper(*args, **kwargs):
        nonlocal obj
        if obj is None:
            obj = cls(*args, **kwargs)
        return obj

    return wrapper


@singleton
@dataclass
class Config:
    image_dir: str = field(default="")
    artifact_prob: float = field(default=0.5)
    noise_coef: float = field(default=0.5)
    distribution: str = field(default="Gaussian")
    kernel_size: int = field(default=5)
    blur_type: str = field(default="Gaussian")
    brightness: float = field(default=0)
    noise_enabled: bool = field(default=True)
    blur_enabled: bool = field(default=True)
    light_enabled: bool = field(default=True)

    def __post_init__(self):
        self.sampled_images = sample_images(self.image_dir)


def visualize(
    tensors,
    n_col=None,
    max_rows=np.inf,
    row_captions=None,
    save_to=None,
    height=5,
    width=5,
    title="",
):
    if n_col is None:
        n_col = min(len(tensors) // 2, 3)

    n_row = len(tensors) // n_col
    assert n_row * n_col == tensors.shape[0]
    if row_captions:
        assert len(row_captions) == n_row

    plt.figure(figsize=(n_col * width, n_row * height))
    plt.title(title, fontsize=10)
    plt.axis("off")

    for i in range(n_row):
        if i + 1 > max_rows:
            break
        for j in range(n_col):
            idx = j + i * n_col
            plt.subplot(n_row, n_col, idx + 1)
            plt.imshow(tensors[idx].permute(1, 2, 0))
            if row_captions:
                plt.title(row_captions[i])
            plt.axis("off")

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")

    plt.show()


def read_image(path=None):
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    if path is not None:
        img = Image.open(path)
    img = transforms(img)
    return img


def sample_images(image_folder, n_images=9):
    image_pathes = os.listdir(image_folder)
    indexes = random.sample(range(0, len(image_pathes) - 1), n_images)
    images = []
    for idx in indexes:
        path = os.path.join(image_folder, image_pathes[idx])
        image = read_image(path)
        images.append(image)

    images = torch.stack(images)
    return images


def make_dirs(destination, sub_dirs=None):
    """Build a dir tree for further train-test split"""
    os.makedirs(destination, exist_ok=True)
    shutil.rmtree(destination)
    os.mkdir(destination)
    if sub_dirs is None:
        sub_dirs = ["test", "train"]
    for dir_name in sub_dirs:
        absolute_path = os.path.join(destination, dir_name)
        try:
            os.mkdir(absolute_path)
        except FileExistsError:
            shutil.rmtree(absolute_path)
            os.mkdir(absolute_path)


def split_images(
    source, destination, train_size=None, test_size=None, paired_dataset=False
):
    """Split images between train and test samples and move them to their corresponding dirs"""
    assert train_size is not None or test_size is not None, "Unspecified split fraction"
    train_size = train_size if train_size is not None else 1 - test_size
    images = sorted(glob(source + "/*.*"))
    train_images, test_images = train_test_split(images, train_size=train_size)

    sub_dirs = (
        ["train", "test"]
        if not paired_dataset
        else ["test_input", "test_output", "train_input", "train_output"]
    )
    make_dirs(destination, sub_dirs)

    for img_path in train_images:
        dst_path = os.path.join(
            destination,
            "train" if not paired_dataset else "train_output",
            os.path.basename(img_path),
        )
        shutil.copyfile(img_path, dst_path)

    for img_path in test_images:
        dst_path = os.path.join(
            destination,
            "test" if not paired_dataset else "test_output",
            os.path.basename(img_path),
        )
        shutil.copyfile(img_path, dst_path)

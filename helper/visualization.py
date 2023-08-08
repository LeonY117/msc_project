import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from torchvision import transforms
from torchvision.utils import make_grid


class ImageDisplayer:
    def __init__(self, images, labels, predictions=None):
        self.images = images
        self.labels = labels
        self.predictions = predictions

    def show_sample(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self.images))
        if len(self.images.shape) == 3:
            self.images = self.images.unsqueeze(0)
            self.labels = self.labels.unsqueeze(0)

        img = self.images[idx].permute(1, 2, 0)
        # labels = self.labels[idx].permute(1, 2, 0)
        labels = self.labels[idx].squeeze()

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("True Mask")
        plt.imshow(labels)
        plt.axis("off")

        plt.tight_layout()


def show_features(features, num_images=20, offset=0):
    i = offset
    n = num_images
    plt.figure(figsize=(20, 4))
    plt.axis("off")
    plt.imshow(
        make_grid(features.detach().permute(1, 0, 2, 3)[i : n + i], nrow=10).permute(
            1, 2, 0
        )
    )


def unnormalize(image: torch.Tensor) -> torch.Tensor:
    """
    Reverses imageNet Normalization to [0, 1], (for visualization purposes)
    """
    mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
    reverse_normalize = transforms.Normalize(mean, std)

    return torch.clip(reverse_normalize(image), 0, 1)

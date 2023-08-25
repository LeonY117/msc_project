import random
import math
import torch
from torch import is_tensor
from torchvision import transforms

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


class HFlip(object):
    def __init__(self, prob: Optional[float] = 0.5) -> None:
        self.flip_transform = transforms.RandomHorizontalFlip(1)
        self.prob = prob

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, y) = data
        p = random.random()
        if p < self.prob:
            x = self.flip_transform(x)
            y = self.flip_transform(y)

        return (x, y)


class HueSatCon(object):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        prob: Optional[float] = 1.0,
    ) -> None:
        self.hue_transform = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.prob = prob

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, y) = data
        p = random.random()
        if p < self.prob and x != None:
            x = self.hue_transform(x)
        return (x, y)


class Rotation(object):
    def __init__(self, angle: int, fill_pix: int, prob: Optional[float] = 1) -> None:
        self.angle = angle
        self.prob = prob
        self.fill_pix = fill_pix

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, y) = data
        p = random.random()
        if p < self.prob:
            # comptute random angle to rotate for this instance
            rot_angle = (random.random() * 2 - 1) * self.angle
            x = transforms.functional.rotate(x, rot_angle, expand=False)
            y = transforms.functional.rotate(
                y, rot_angle, expand=False, fill=self.fill_pix
            )
        return (x, y)


class Crop(object):
    def __init__(self, crop_percentage: float, prob: Optional[float] = 1) -> None:
        self.crop_percentage = crop_percentage
        self.prob = prob

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, y) = data
        p = random.random()
        if p < self.prob:
            # compute random crop proportions for this instance
            if is_tensor(x):
                dim1 = x.shape[1]  # vertical length
                dim2 = x.shape[2]  # horizontal length
            else:
                dim1, dim2 = x.size
            # percentage crop in the vertical direction
            crop1 = random.random() * (1 - self.crop_percentage) + self.crop_percentage
            # percentage crop in the horizontal direction
            crop2 = random.random() * (1 - self.crop_percentage) + self.crop_percentage
            # compute all the arguments for TF.crop()
            height = int(crop1 * dim1)
            width = int(crop2 * dim2)
            top = int(random.random() * ((1 - crop1) / 2) * dim1)
            left = int(random.random() * ((1 - crop2) / 2) * dim2)

            x = transforms.functional.crop(x, top, left, height, width)
            y = transforms.functional.crop(y, top, left, height, width)

        return (x, y)


# Translation and Shear
class Affine(object):
    def __init__(
        self,
        affine_percentage: float,
        shear_deg: int,
        fill_pix: int,
        scale: Optional[Tuple[float, float]] = [1.0, 1.0],
        prob: Optional[float] = 1.0,
    ) -> None:
        self.affine_percentage = affine_percentage  # (e.g. 0.1)
        self.shear_deg = shear_deg  # in deg (e.g. 15)
        self.fill_pix = fill_pix
        self.scale = scale
        self.prob = prob

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, y) = data

        p = random.random()
        if p < self.prob:
            # compute random affine proportions for this instance
            if is_tensor(x):
                dim1 = x.shape[1]  # horizontal length
                dim2 = x.shape[2]  # vertical length
            else:
                dim1, dim2 = x.size

            # translation offset for this instance
            offset1 = (random.random() * 2 - 1) * dim1 * self.affine_percentage
            offset2 = (random.random() * 2 - 1) * dim2 * self.affine_percentage
            # shear deg for this instance
            shear = (random.random() * 2 - 1) * self.shear_deg
            scale = (random.random() * 2 - 1) * (
                self.scale[1] - self.scale[0]
            ) + self.scale[0]

            # apply affine transformation
            x = transforms.functional.affine(
                x,
                angle=0,
                translate=[offset1, offset2],
                shear=[shear, 0],
                scale=scale,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            y = transforms.functional.affine(
                y,
                angle=0,
                translate=[offset1, offset2],
                shear=[shear, 0],
                scale=scale,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=self.fill_pix,
            )
        return (x, y)


class Resize(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self.resize_x = transforms.Resize(
            size, transforms.InterpolationMode.BILINEAR, antialias=True
        )
        self.resize_y = transforms.Resize(
            size, transforms.InterpolationMode.NEAREST, antialias=True
        )  # nearest-exact?

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = data
        x = self.resize_x(x)
        y = self.resize_y(y)

        return (x, y)


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.PILToTensor()

    def __call__(self, data):
        x, y = data

        x = self.to_tensor(x)
        y = self.to_tensor(y)

        return (x, y)


class Normalize(object):
    def __init__(
        self, drop_pix: Optional[int] = None, fill_pix: Optional[int] = None
    ) -> None:
        # imageNet Normalize
        self.Normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.drop_pix = drop_pix
        self.fill_pix = fill_pix

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = data
        x = self.Normalize(x)
        if self.drop_pix is not None:
            y = torch.where(y == self.drop_pix, torch.ones_like(y) * self.fill_pix, y)

        return (x, y)


class RandomResizedCrop(torch.nn.Module):
    """Crop the given image to random size and aspect ratio.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
            In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): scale range of the cropped image before resizing, relatively to the origin image.
        ratio (tuple of float): aspect ratio range of the cropped image before resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.5, 2.0), ratio=(1, 1)):
        super().__init__()
        self.size = size

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: torch.Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = transforms.functional.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(
        self, data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        (x, y) = data

        i, j, h, w = self.get_params(x, self.scale, self.ratio)
        x = transforms.functional.resized_crop(
            x,
            i,
            j,
            h,
            w,
            self.size,
            transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        y = transforms.functional.resized_crop(
            y,
            i,
            j,
            h,
            w,
            self.size,
            transforms.InterpolationMode.NEAREST,
            antialias=False,
        )
        return (x, y)

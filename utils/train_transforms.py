import random
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
        prob: Optional[float] = 1.0,
    ) -> None:
        self.affine_percentage = affine_percentage  # (e.g. 0.1)
        self.shear_deg = shear_deg  # in deg (e.g. 15)
        self.fill_pix = fill_pix
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

            # apply affine transformation
            x = transforms.functional.affine(
                x,
                angle=0,
                translate=[offset1, offset2],
                shear=[shear, 0],
                scale=1,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            y = transforms.functional.affine(
                y,
                angle=0,
                translate=[offset1, offset2],
                shear=[shear, 0],
                scale=1,
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

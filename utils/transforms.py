import torch
import random
from torchvision import transforms

# Flip, Hue & sat, Rotation, Scaling, Resize, Normalize

# Only on tensors!
class HFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.HFlip = transforms.RandomHorizontalFlip(1)

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        p = random.random()
        if p < self.prob:
            image = self.HFlip(image)
            mask = self.HFlip(mask)

        return {'image': image, 'mask': mask}



class HueSatCon(object):
    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.prob = prob
        self.hue_transform = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        p = random.random()
        if p < self.prob:
            image = self.hue_transform(image)
        return {'image': image, 'mask': mask}


class Rotation(object):
    def __init__(self, angle, prob=1):
        self.prob = prob
        self.angle = angle

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        rot_angle = (random.random()*2-1) * self.angle
        p = random.random()
        if p < self.prob:
            image = transforms.functional.rotate(image, rot_angle, expand=True)
            mask = transforms.functional.rotate(mask, rot_angle, expand=True, fill=3)
        return {'image': image, 'mask': mask}

class Crop(object):
    def __init__(self, crop_percentage):
        self.crop_percentage = crop_percentage
    
    def __call__(self, data):
        image, mask = data['image'], data['mask']
        img_dim = image.shape[1]
        height = int(self.crop_percentage * img_dim)
        width = int(self.crop_percentage * img_dim)
        left = int(random.random() * ((1-self.crop_percentage)/2) * img_dim)
        top = int(random.random() * ((1-self.crop_percentage)/2) * img_dim)

        image = transforms.functional.crop(image, top, left, height, width)
        mask = transforms.functional.crop(mask, top, left, height, width)

        return {'image': image, 'mask': mask}

# class Resize(object):
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, data):
#         image, mask = data['image'], data['mask']
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         image = transform.resize(image, (new_h, new_w))
#         mask = transform.resize(mask, (new_h, new_w))

#         return {'image': image, 'mask': mask}

class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize_image = transforms.Resize(size, transforms.InterpolationMode.BILINEAR) 
        self.resize_mask = transforms.Resize(size, transforms.InterpolationMode.NEAREST) #nearest-exact?

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        image = self.resize_image(image)
        mask = self.resize_mask(mask)

        return {'image': image, 'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0 # normalize
        mask = torch.from_numpy(mask).unsqueeze(0).to(dtype=torch.float32)
        mask = torch.where(mask == 4, torch.ones_like(mask)*3, mask)
        return {'image': image, 'mask': mask}
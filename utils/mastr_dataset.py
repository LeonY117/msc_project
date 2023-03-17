import os
# from skimage import io
from torchvision.io import read_image
from torch.utils.data import Dataset


class MastrDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform

        self.img_file_names = os.listdir(img_dir)

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_file_names[idx])
        mask_path = os.path.join(
            self.mask_dir, self.img_file_names[idx].replace(".jpg", "m.png"))

        # image = io.imread(img_path)
        # mask = io.imread(mask_path)

        # NOTE: image format: (C, W, H)

        image = read_image(img_path)
        mask = read_image(mask_path)

        # NOTE: image format(C, W, H)

        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        return sample

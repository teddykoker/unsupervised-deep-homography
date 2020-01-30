import torch
import random
from pathlib import Path
import kornia
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SyntheticDataset(Dataset):
    def __init__(self, folder, patch_size=128, rho=32):
        super(SyntheticDataset, self).__init__()
        self.fnames = list(Path(folder).glob("*.jpg"))
        self.transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )
        self.patch_size = patch_size
        self.rho = rho

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_a = Image.open(self.fnames[index])
        img_a = self.transforms(img_a)

        # pick top left corner
        x = random.randint(self.rho, 256 - self.rho - self.patch_size)
        y = random.randint(self.rho, 256 - self.rho - self.patch_size)

        points = torch.tensor(
            [
                [x, y],
                [x + self.patch_size, y],
                [x + self.patch_size, y + self.patch_size],
                [x, y + self.patch_size],
            ]
        )
        perturbed_points = points + torch.randint_like(points, -self.rho, self.rho)

        # compute homography from points
        h = kornia.get_perspective_transform(
            points.unsqueeze(0).float(), perturbed_points.unsqueeze(0).float()
        )

        # apply homography to single img
        img_b = kornia.warp_perspective(img_a.unsqueeze(0), h, (256, 256))[0]

        patch_a = img_a[:, y : y + self.patch_size, x : x + self.patch_size]
        patch_b = img_b[:, y : y + self.patch_size, x : x + self.patch_size]

        return img_a, img_b, patch_a, patch_b, points.float()

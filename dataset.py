import torch
import random
from pathlib import Path
import kornia
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

MEAN = torch.tensor([0.485, 0.456, 0.406]).mean().unsqueeze(0)
STD = torch.tensor([0.229, 0.224, 0.225]).mean().unsqueeze(0)


def safe_collate(batch):
    """Return batch without any None values"""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class SyntheticDataset(Dataset):
    def __init__(self, folder, filetype=".jpg", patch_size=128, rho=45):
        super(SyntheticDataset, self).__init__()
        self.fnames = list(Path(folder).glob(f"*{filetype}"))
        self.transforms = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        self.patch_size = patch_size
        self.rho = rho

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_a = Image.open(self.fnames[index])
        img_a = self.transforms(img_a)

        # grayscale
        img_a = torch.mean(img_a, dim=0, keepdim=True)

        # pick top left corner
        x = random.randint(self.rho, 256 - self.rho - self.patch_size)
        y = random.randint(self.rho, 256 - self.rho - self.patch_size)

        corners = torch.tensor(
            [
                [x, y],
                [x + self.patch_size, y],
                [x + self.patch_size, y + self.patch_size],
                [x, y + self.patch_size],
            ]
        )
        delta = torch.randint_like(corners, -self.rho, self.rho)
        perturbed_corners = corners + delta

        try:
            # compute homography from points
            h = kornia.get_perspective_transform(
                corners.unsqueeze(0).float(), perturbed_corners.unsqueeze(0).float()
            )

            h_inv = torch.inverse(h)

            # apply homography to single img
            img_b = kornia.warp_perspective(img_a.unsqueeze(0), h_inv, (256, 256))[0]

        except:
            # either matrix could not be solved or inverted
            # this will show up as None, so use safe_collate in train.py
            return

        patch_a = img_a[:, y : y + self.patch_size, x : x + self.patch_size]
        patch_b = img_b[:, y : y + self.patch_size, x : x + self.patch_size]

        return img_a, patch_a, patch_b, corners.float(), delta.float()

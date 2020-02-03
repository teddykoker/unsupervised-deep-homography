import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import kornia
import argparse
from pathlib import Path

from dataset import SyntheticDataset
from model import Net


def photometric_loss(delta, img_a, patch_b, corners):
    corners_hat = corners + delta

    # in order to apply transform and center crop,
    # subtract points by top-left corner (corners[N, 0])
    corners = corners - corners[:, 0].view(-1, 1, 2)

    h = kornia.get_perspective_transform(corners, corners_hat)

    h_inv = torch.inverse(h)
    patch_b_hat = kornia.warp_perspective(img_a, h_inv, (128, 128))

    # smooth l1 loss being used in tensorflow implementation
    return F.smooth_l1_loss(patch_b_hat, patch_b)


def train_step(model, optimizer, dataloader, device, writer):
    model.train()
    total_loss = 0.0
    size = len(dataloader.dataset)
    for img_a, patch_a, patch_b, corners in tqdm(dataloader, leave=False):
        img_a, patch_a, patch_b, corners = (
            img_a.to(device),
            patch_a.to(device),
            patch_b.to(device),
            corners.to(device),
        )
        delta = model(patch_a, patch_b)
        loss = photometric_loss(delta, img_a, patch_b, corners)
        total_loss += loss.item() * img_a.size(0)
        loss.backward()
        optimizer.step()
        writer.add_scalar("train_loss", loss.item())
        writer.flush()
    return total_loss / size


def valid_step(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        size = len(dataloader.dataset)
        for img_a, patch_a, patch_b, corners in tqdm(dataloader, leave=False):
            img_a, patch_a, patch_b, corners = (
                img_a.to(device),
                patch_a.to(device),
                patch_b.to(device),
                corners.to(device),
            )
            delta = model(patch_a, patch_b)
            loss = photometric_loss(delta, img_a, patch_b, corners)
            total_loss += loss.item() * img_a.size(0)
        return total_loss / size


def fit(opt):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticDataset(opt.train_path, filetype=opt.filetype)
    train_size = int(0.8 * len(dataset))
    train_set, valid_set = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_loader = DataLoader(train_set, batch_size=opt.bs, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=opt.bs, num_workers=4)

    writer = SummaryWriter()
    model = Net().to(device)

    if opt.resume:
        print("resuming")
        model.load_state_dict(torch.load(opt.model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for e in range(opt.epochs):
        train_loss = train_step(model, optimizer, train_loader, device, writer)
        valid_loss = valid_step(model, valid_loader, device)
        print(f"{e}\t{train_loss:.4f}\t{valid_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/{e}.pt")
    torch.save(model.state_dict(), opt.model_path)


def test_dataset(path):
    dataset = SyntheticDataset(path)

    img_a, img_b, patch_a, patch_b, points = dataset[0]
    print(img_a.shape)
    print(patch_a.shape)
    print(patch_b.shape)
    print(points.shape)

    import matplotlib.pyplot as plt
    from torchvision import transforms

    to_pil = transforms.ToPILImage()

    for img in [img_a, patch_a, patch_b]:
        plt.imshow(to_pil(img), cmap="gray")
        plt.show()


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--filetype", default=".jpg", help="filetype of images")
    parser.add_argument("--model_path", default="models/model.pt", help="path to model")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("train_path", help="path to training data")
    opt = parser.parse_args()
    fit(opt)

# test_dataset()

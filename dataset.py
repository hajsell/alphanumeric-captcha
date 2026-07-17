import os
import glob
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CaptchaDataset(Dataset):
    def __init__(self, img_paths, transform=None, char_to_idx=None):
        self.img_paths = img_paths
        self.transform = transform
        self.char_to_idx = char_to_idx
        if self.char_to_idx is None:
            self.char_to_idx = self._build_char_to_idx()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label_str = os.path.basename(path).split(".")[0].strip()
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = [self.char_to_idx[char] for char in label_str]
        return img, torch.tensor(label, dtype=torch.long)

    def _build_char_to_idx(self):
        unique_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(list(unique_chars)))}
        char_to_idx['-'] = 0
        return char_to_idx


def collate_fn(batch):
    images, labels, target_lengths = [], [], []
    for img, label in batch:
        images.append(img)
        labels.append(label)
        target_lengths.append(len(label))
    return torch.stack(images), torch.cat(labels), torch.tensor(target_lengths, dtype=torch.long)


def get_data_loaders(data_dir="dataset", batch_size=128, num_workers=4, seed=22520467):
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_img_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))

    if not all_img_paths:
        raise FileNotFoundError(f"No images found in target directory: {data_dir}")

    random.seed(seed)
    random.shuffle(all_img_paths)

    train_end = int(len(all_img_paths) * 0.8)
    val_end = train_end + int(len(all_img_paths) * 0.1)

    train_dataset = CaptchaDataset(all_img_paths[:train_end], transform=train_transform)
    char_to_idx = train_dataset.char_to_idx

    val_dataset = CaptchaDataset(all_img_paths[train_end:val_end], transform=val_transform, char_to_idx=char_to_idx)
    test_dataset = CaptchaDataset(all_img_paths[val_end:], transform=val_transform, char_to_idx=char_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, len(char_to_idx), "".join([c for c in char_to_idx.keys() if c != '-'])
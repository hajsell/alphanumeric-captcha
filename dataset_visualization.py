import os
import random
import torch
import matplotlib.pyplot as plt

from dataset import get_data_loaders
from model import CRNN_ResNet


def main():
    # Configuration
    BATCH_SIZE = 128
    NUM_SAMPLES_TO_PLOT = 12
    OUTPUT_PNG = "dataset_raw_samples.png"
    DPI = 300

    # Load test dataloader and alphabet parameters
    _, _, test_loader, vocab_size, chars = get_data_loaders(
        data_dir="dataset",
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    # Initialize model to access the vocabulary mapping (idx2char)
    model = CRNN_ResNet(vocab_size=vocab_size, chars=chars)
    char_map = model.idx2char

    # Extract a single batch of validation/test data
    images, labels, target_lengths = next(iter(test_loader))

    # ImageNet normalization parameters for correct RGB visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Decode target labels using model character map
    true_labels = []
    current_idx = 0

    for length in target_lengths:
        seq = labels[current_idx : current_idx + length]
        label_str = "".join([char_map[idx.item()] for idx in seq])
        true_labels.append(label_str)
        current_idx += length

    # Select random indices for a 3x4 visualization grid
    indices = random.sample(range(len(images)), NUM_SAMPLES_TO_PLOT)

    # Generate the visualization plot (3 rows, 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(12, 6.5))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Denormalize image tensor to [0, 1] range
        img_tensor = images[idx] * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_numpy = img_tensor.permute(1, 2, 0).numpy()

        # Render image with its ground truth label
        ax.imshow(img_numpy)
        ax.set_title(f"Label: {true_labels[idx]}", fontsize=11, fontweight="bold", color="black", pad=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Dataset preview collage generated successfully. Output saved to: '{OUTPUT_PNG}'")


if __name__ == "__main__":
    main()
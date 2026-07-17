import os
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_data_loaders
from model import CRNN_ResNet


def main():
    # Configuration
    MODEL_PATH = "crnn_resnet_best.pth"
    BATCH_SIZE = 128
    NUM_SAMPLES_TO_POOL = 20
    NUM_SAMPLES_TO_PLOT = 4
    OUTPUT_IMAGE_PATH = "analytics_prediction_samples.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataloader (without augmentations)
    _, _, test_loader, vocab_size, chars = get_data_loaders(
        data_dir="dataset",
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    # Initialize model
    model = CRNN_ResNet(vocab_size=vocab_size, chars=chars).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Target checkpoint not found at '{MODEL_PATH}'")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    correct_samples = []
    incorrect_samples = []

    print("Analyzing test dataset for qualitative samples...")

    # ImageNet denormalization parameters for correct matplotlib visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    with torch.no_grad():
        for images, labels, target_lengths in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            logits = model(images)
            predictions = model.decode(logits)

            true_labels = []
            current_idx = 0
            for length in target_lengths:
                seq = labels[current_idx: current_idx + length]
                label_str = "".join([model.idx2char[idx.item()] for idx in seq])
                true_labels.append(label_str)
                current_idx += length

            for i, (pred, true) in enumerate(zip(predictions, true_labels)):
                # Denormalize image tensor to [0, 1] range
                img_tensor = images[i] * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_numpy = img_tensor.cpu().permute(1, 2, 0).numpy()

                if pred == true:
                    if len(correct_samples) < NUM_SAMPLES_TO_POOL:
                        correct_samples.append((img_numpy, true, pred))
                else:
                    if len(incorrect_samples) < NUM_SAMPLES_TO_POOL:
                        incorrect_samples.append((img_numpy, true, pred))

            if len(correct_samples) >= NUM_SAMPLES_TO_POOL and len(incorrect_samples) >= NUM_SAMPLES_TO_POOL:
                break

    # Randomly select samples for plotting
    selected_correct = random.sample(correct_samples, min(NUM_SAMPLES_TO_PLOT, len(correct_samples)))
    selected_incorrect = random.sample(incorrect_samples, min(NUM_SAMPLES_TO_PLOT, len(incorrect_samples)))

    # Generate the visualization plot (4 rows, 2 columns)
    fig, axes = plt.subplots(NUM_SAMPLES_TO_PLOT, 2, figsize=(10, 10))

    # Column 1: Correct predictions
    for row, (img, true, pred) in enumerate(selected_correct):
        ax = axes[row, 0]
        ax.imshow(img)
        ax.set_title(f"Label: {true} | Prediction: {pred}", color="forestgreen", fontsize=11, fontweight="bold")
        ax.axis("off")
        if row == 0:
            ax.text(0.5, 1.3, "Correct Predictions", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, fontweight="bold", color="black")

    # Column 2: Incorrect predictions
    for row, (img, true, pred) in enumerate(selected_incorrect):
        ax = axes[row, 1]
        ax.imshow(img)
        display_pred = pred if pred.strip() != "" else "[empty]"
        ax.set_title(f"Label: {true} | Prediction: {display_pred}", color="crimson", fontsize=11, fontweight="bold")
        ax.axis("off")
        if row == 0:
            ax.text(0.5, 1.3, "Incorrect Predictions", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, fontweight="bold", color="black")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Qualitative analysis collage generated successfully. Output saved to: '{OUTPUT_IMAGE_PATH}'")


if __name__ == "__main__":
    main()
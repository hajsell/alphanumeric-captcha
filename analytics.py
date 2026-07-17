import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_data_loaders
from model import CRNN_ResNet


def macro_classify(char):
    if char.isdigit():
        return 'Digits'
    elif char.islower():
        return 'Lowercase'
    elif char.isupper():
        return 'Uppercase'
    return 'Unknown'


def main():
    # Configuration
    MODEL_PATH = "crnn_resnet_best.pth"
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    DPI = 300

    CHAR_PERFORMANCE_PNG = "analytics_char_performance.png"
    MACRO_CONFUSION_PNG = "analytics_macro_confusion_matrix.png"
    FILTERED_CONFUSION_PNG = "analytics_filtered_confusion_matrix.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataloader
    _, _, test_loader, vocab_size, chars = get_data_loaders(
        data_dir="dataset",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Initialize model
    model = CRNN_ResNet(vocab_size=vocab_size, chars=chars).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Target checkpoint not found at '{MODEL_PATH}'")
        return

    # Load state dict with weights_only=True to prevent FutureWarning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    all_preds_chars = []
    all_true_chars = []

    print("Executing model predictions on isolated test dataset...")
    with torch.no_grad():
        for images, labels, target_lengths in tqdm(test_loader, desc="Test Inference"):
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

            for pred, true in zip(predictions, true_labels):
                # Align lengths to ensure identical array sizes
                min_len = min(len(pred), len(true))
                for i in range(min_len):
                    all_preds_chars.append(pred[i])
                    all_true_chars.append(true[i])

    unique_labels = sorted(list(set(all_true_chars + all_preds_chars)))
    if '-' in unique_labels:
        unique_labels.remove('-')

    # Section 1: Classification Report & F1-Score Plotting (Best vs. Worst)
    report = classification_report(all_true_chars, all_preds_chars, output_dict=True, zero_division=0)

    char_metrics = []
    for char in unique_labels:
        if char in model.char2idx and char != '-':
            char_metrics.append((char, report[char]['f1-score']))

    char_metrics.sort(key=lambda x: x[1])
    worst_10 = char_metrics[:10]
    best_10 = char_metrics[-10:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.barh([x[0] for x in worst_10], [x[1] for x in worst_10], color='crimson')
    ax1.set_title("Top 10 Lowest Performing Characters (F1-Score)", fontsize=12)
    ax1.set_xlabel("F1-Score", fontsize=10)
    ax1.set_xlim(0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.barh([x[0] for x in best_10], [x[1] for x in best_10], color='forestgreen')
    ax2.set_title("Top 10 Highest Performing Characters (F1-Score)", fontsize=12)
    ax2.set_xlabel("F1-Score", fontsize=10)
    ax2.set_xlim(0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CHAR_PERFORMANCE_PNG, dpi=DPI)
    plt.close()

    # Section 2: Macro-Class Confusion Matrix (3x3 Aggregation)
    macro_true = [macro_classify(c) for c in all_true_chars]
    macro_pred = [macro_classify(c) for c in all_preds_chars]
    macro_categories = ['Digits', 'Lowercase', 'Uppercase']

    cm_macro = confusion_matrix(macro_true, macro_pred, labels=macro_categories)
    cm_macro_percent = cm_macro.astype('float') / cm_macro.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_macro_percent, annot=cm_macro, fmt="d", cmap="Blues",
                xticklabels=macro_categories, yticklabels=macro_categories, cbar=False,
                annot_kws={"size": 11})

    for i in range(len(macro_categories)):
        for j in range(len(macro_categories)):
            plt.gca().text(j + 0.5, i + 0.7, f"({cm_macro_percent[i, j]:.1f}%)",
                           ha='center', va='center', color='black', fontsize=9)

    plt.title("Macro-Class Aggregated Confusion Matrix (Percentage Scaling)", fontsize=12)
    plt.ylabel("True Macro-Class", fontsize=10)
    plt.xlabel("Predicted Macro-Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(MACRO_CONFUSION_PNG, dpi=DPI)
    plt.close()

    # Section 3: Filtered Confusion Matrix (Most Problematic Pairs Only)
    cm_full = confusion_matrix(all_true_chars, all_preds_chars, labels=unique_labels)
    np.fill_diagonal(cm_full, 0)

    # Filter out weak relationships to focus on high-frequency error pairs
    error_threshold = sorted(cm_full.flatten(), reverse=True)[20]
    problematic_indices = np.where(cm_full >= max(error_threshold, 1))

    problem_chars = sorted(list(set([unique_labels[i] for i in problematic_indices[0]] +
                                    [unique_labels[j] for j in problematic_indices[1]])))

    if '-' in problem_chars:
        problem_chars.remove('-')

    if problem_chars:
        cm_filtered = confusion_matrix(all_true_chars, all_preds_chars, labels=problem_chars)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_filtered, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=problem_chars, yticklabels=problem_chars, cbar=True)
        plt.title("Filtered Confusion Matrix (Most Misclassified Character Pairs)", fontsize=12)
        plt.ylabel("True Label", fontsize=10)
        plt.xlabel("Predicted Label", fontsize=10)
        plt.tight_layout()
        plt.savefig(FILTERED_CONFUSION_PNG, dpi=DPI)
        plt.close()

    print("Analytical processing complete. Statistical visualizations successfully generated.")


if __name__ == "__main__":
    main()
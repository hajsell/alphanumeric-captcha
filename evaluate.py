import os
import pickle as pkl
import matplotlib.pyplot as plt


def generate_academic_plots(history_path="training_history.pkl"):
    # Plotting configuration
    GRID_STYLE = "--"
    GRID_ALPHA = 0.6
    LINE_WIDTH = 1.5
    DPI = 300

    if not os.path.exists(history_path):
        print(f"Error: Target metrics record not found at path: '{history_path}'")
        return

    with open(history_path, "rb") as f:
        history = pkl.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    val_loss_key = "history_val_loss" if "history_val_loss" in history else "val_loss"

    # Plot 1: Loss Convergence (CTC Loss Curve)
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
        color="blue",
        linewidth=LINE_WIDTH
    )
    plt.plot(
        epochs,
        history[val_loss_key],
        label="Validation Loss",
        color="orange",
        linewidth=LINE_WIDTH
    )
    plt.title("Objective Function Convergence Analysis (CTC Loss)", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss Value", fontsize=10)
    plt.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("metric_loss_convergence.png", dpi=DPI)
    plt.close()

    # Plot 2: Accuracy Evaluation (Sequence & Character Level)
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        history["val_char_acc"],
        label="Character-Level Accuracy",
        color="green",
        linewidth=LINE_WIDTH
    )
    plt.plot(
        epochs,
        history["val_word_acc"],
        label="Sequence-Level Accuracy (Full CAPTCHA)",
        color="red",
        linewidth=LINE_WIDTH
    )
    plt.title("Model Recognition Accuracy Growth Profile", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy Percentage (%)", fontsize=10)
    plt.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("metric_accuracy_growth.png", dpi=DPI)
    plt.close()

    print("Academic metrics plotting procedure complete. Plot binaries successfully saved.")


if __name__ == "__main__":
    generate_academic_plots()
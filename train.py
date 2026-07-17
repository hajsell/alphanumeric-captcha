import os
import sys
import time
import pickle as pkl
import torch
from tqdm import tqdm

from dataset import get_data_loaders
from model import CRNN_ResNet


def main():
    # Configuration
    MODEL_PATH = "crnn_resnet_best.pth"
    HISTORY_PATH = "training_history.pkl"
    EPOCHS = 30
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    GRADIENT_CLIP_VAL = 5.0
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("CRITICAL ERROR: CUDA execution environment is unavailable.")
        print("Please check your NVIDIA drivers, CUDA Toolkit installation, or PyTorch build configuration.")
        sys.exit(1)

    print(f"Execution target verified successfully. Device: {torch.cuda.get_device_name(0)}")

    # Load data loaders
    train_loader, val_loader, _, vocab_size, chars = get_data_loaders(
        data_dir="dataset",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Initialize model, optimizer, and learning rate scheduler
    model = CRNN_ResNet(vocab_size=vocab_size, chars=chars).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Load or initialize training history log
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pkl.load(f)
        print(f"History log located. Active record contains {len(history['train_loss'])} epoch entries.")
    else:
        history = {"train_loss": [], "val_loss": [], "val_char_acc": [], "val_word_acc": []}

    # Load checkpoint if exists and evaluate starting validation loss
    if os.path.exists(MODEL_PATH):
        print(f"Weights checkpoint found. Loading state dict from: '{MODEL_PATH}'")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        initial_val_loss = 0.0
        with torch.no_grad():
            for images, labels, target_lengths in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.int32).to(device)
                loss = model.criterion(logits.log_softmax(2), labels, input_lengths, target_lengths)
                initial_val_loss += loss.item()
        best_val_loss = initial_val_loss / len(val_loader)
        print(f"Resuming pipeline execution with base Val Loss: {best_val_loss:.4f}")
    else:
        print("Initialization complete. Executing training from fundamental state.")
        best_val_loss = float('inf')

    # Main training loop
    for epoch in range(EPOCHS):
        start_time = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=False)
        for images, labels, target_lengths in pbar_train:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.int32).to(device)
            loss = model.criterion(logits.log_softmax(2), labels, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optimizer.step()

            train_loss += loss.item()
            pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase & Accuracy Metrics Calculation
        model.eval()
        val_loss = 0.0
        total_characters = 0
        correct_characters = 0
        total_words = 0
        correct_words = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for images, labels, target_lengths in pbar_val:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.int32).to(device)
                loss = model.criterion(logits.log_softmax(2), labels, input_lengths, target_lengths)
                val_loss += loss.item()

                predictions = model.decode(logits)
                true_labels = []
                current_idx = 0
                for length in target_lengths:
                    seq = labels[current_idx: current_idx + length]
                    label_str = "".join([model.idx2char[idx.item()] for idx in seq])
                    true_labels.append(label_str)
                    current_idx += length

                for pred, true in zip(predictions, true_labels):
                    total_words += 1
                    if pred == true:
                        correct_words += 1

                    min_len = min(len(pred), len(true))
                    for p_char, t_char in zip(pred[:min_len], true[:min_len]):
                        if p_char == t_char:
                            correct_characters += 1
                    total_characters += len(true)

                pbar_val.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        epoch_char_acc = (correct_characters / total_characters) * 100
        epoch_word_acc = (correct_words / total_words) * 100
        epoch_time = time.time() - start_time

        # Save metrics to history dictionary
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_char_acc"].append(epoch_char_acc)
        history["val_word_acc"].append(epoch_word_acc)

        with open(HISTORY_PATH, "wb") as f:
            pkl.dump(history, f)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Char Acc: {epoch_char_acc:.2f}% | "
              f"Word Acc: {epoch_word_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best model checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new optimal checkpoint to local storage: '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
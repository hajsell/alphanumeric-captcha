# CAPTCHA Solver: End-to-End Sequence Recognition with CRNN

This repository contains an end-to-end deep learning model for recognizing multidigit alphanumeric CAPTCHA sequences without prior character segmentation.

## 🚀 Key Features
- **Hybrid Architecture:** Combines a convolutional feature extractor (ResNet-18) with a sequence model (Bidirectional LSTM)[cite: 1].
- **Segmentation-Free:** Trained end-to-end using **Connectionist Temporal Classification (CTC) Loss**[cite: 1].
- **Critically Tuned:** Stride modified in ResNet layers 3 & 4 to retain sufficient feature map width for CTC alignment[cite: 1].

## 📊 Results
- **Character Accuracy:** ~90.03%[cite: 1]
- **Sequence Accuracy:** ~72.46%[cite: 1]

## 🛠️ How to run
1. Install requirements: `pip install -r requirements.txt`
2. Run inference: `python predict.py --image path/to/captcha.png`
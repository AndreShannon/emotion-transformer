# CNN + Vision Transformer for Facial Emotion Recognition

This project implements a **Vision Transformer (ViT)** and a **CNN + ViT hybrid** in PyTorch to classify 48×48 grayscale face images into 7 emotions:

> angry, disgusted, fearful, happy, neutral, sad, surprised

The dataset is the FER-style **Emotion Detection FER** dataset from Kaggle, and training is done on a Slurm GPU cluster.

ChatGPT was used in this project

---

## Project Overview

### Architectures

1. **VisionTransformerEmotion (patch-based ViT)**  
   - Patch size: 4 or 8  
   - Embedding dim: 128–192  
   - Depth: 4–6 Transformer encoder blocks  
   - Multi-head self-attention (4–6 heads)  
   - MLP hidden dim: 512–768  
   - Learnable `[CLS]` token and positional embeddings  
   - Classification head: small MLP → 7 logits

2. **VisionTransformerEmotionCNN (CNN + ViT hybrid)**  
   This is the best-performing model.
   - CNN stem:
     - Conv(1→32, 3×3) + BN + ReLU + MaxPool
     - Conv(32→64, 3×3) + BN + ReLU + MaxPool  
     - Output feature map: 64×12×12
   - Flatten spatial grid (12×12 = 144 tokens), project 64→192  
   - Add `[CLS]` token + learnable positional embeddings  
   - Transformer encoder:
     - embed_dim = 192  
     - depth = 6  
     - num_heads = 6  
     - mlp_dim = 768  
     - dropout ≈ 0.15–0.2  
   - Classification head: MLP(192→192→7)

With stronger data augmentation (flip, rotation, slight affine, color jitter, random erasing) and label smoothing, the CNN+ViT model reaches validation accuracy in the **low 60% range** on this dataset.

---

## Dataset

The project uses the **emotion-detection-fer** dataset from Kaggle.

Folder structure (after download/unzip) is expected to be:

```text
data/emotion_detection_fer/
    train/
        angry/
        disgusted/
        fearful/
        happy/
        neutral/
        sad/
        surprised/
    test/
        angry/
        disgusted/
        fearful/
        happy/
        neutral/
        sad/
        surprised/

By default, the script:
- Uses train/ for train + validation (90% / 10% split).
- Uses test/ as the final hold-out test set.

---

## Installation

# Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install requirements.txt

If you use CUDA wheels, adjust the PyTorch install command according to your GPU/driver.

---

## Training

The main training script is: python train_emotion_vit.py

Key features:

- Loads the Kaggle dataset from data/emotion_detection_fer/.
- Applies strong data augmentation on the training split only:
  - Grayscale + resize
  - Random horizontal flip
  - Small rotations / affine transforms
  - Random resized crop
  - Brightness/contrast jitter
  - Random erasing
- Uses a TransformedSubset wrapper so validation and test use only deterministic transforms.
- Supports two model types via run_config["model_type"]:
  - "vit_patch" – pure patch-based ViT
  - "vit_cnn" – CNN + ViT hybrid (default / best model)
- Optimizer: AdamW
- Scheduler: cosine decay with warmup
- Loss: CrossEntropyLoss with optional label smoothing and (optionally) class weights.
- Tracks per-epoch train/val loss and accuracy.

---

## Experiment Tracking

Each training run creates a timestamped directory under runs/, e.g.:
runs/2025xxxx_xxxxxx_vit_cnn_ed192_d6_h6_lr0.0005/
    config.json           # hyperparameters for the run
    best_model.pt         # best validation checkpoint
    training_metrics.csv  # loss/accuracy per epoch
    loss_curve.png        # train vs val loss
    accuracy_curve.png    # train vs val accuracy
    confusion_matrix.png  # normalized confusion matrix with labels
    results.json          # final test metrics (acc, precision, recall, F1, etc)

---

## Metrics & Evaluation

After training, the script:
- Loads the best model (by validation accuracy).
- Evaluates on the test set:
  - Accuracy
  - Per-class precision, recall, and F1
  - Macro and weighted averages
- Plots
  - Normalized confusion matrix 
  - Accuracy on training and validation set per epoch
  - Loss on training and validation set per epoch

---

## Future Work

Ideas for extension:
- Try different CNN stems or deeper ViTs.
- Experiment with class-weighted loss or a weighted sampler to better handle the class imbalance (especially disgusted).
- Add support for loading pre-trained ViT backbones and fine-tuning on this dataset.




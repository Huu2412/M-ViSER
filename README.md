# M-ViSER: Multimodal Speech Emotion Recognition with Logit-Guided Hallucination

M-ViSER is a state-of-the-art Speech Emotion Recognition (SER) system that integrates **MTL-SER (CTC Student ASR)** and **AURORA (Teacher Cross-Modal Distillation)**. It introduces a novel **Logit-Guided Hallucination Architecture**, enabling the model to achieve multimodal-level performance using *audio-only* input during inference. It natively supports standard English benchmarking on IEMOCAP.

## 🚀 Key Features & Architecture

- **Acoustic Backbone (Wav2Vec2)**: Robust acoustic feature extraction with partial fine-tuning. A CTC head performs an auxiliary ASR task to learn phonetic representations.
- **Logit-Guided Attention Pooling**: The student model uses its own CTC logits as a "phonetic highlighter" to dynamically weight audio frames, amplifying speech-rich segments and suppressing silence/noise.
- **Hallucination MLP (Student Path)**: During inference, the student relies on an end-to-end audio-only path. It "hallucinates" (imagines) cross-modal representations without requiring any text input, retaining robust semantic understanding purely from audio.
- **Cross-Modal Distillation (Teacher Path)**: During training, a Teacher path uses Ground-truth text (via BERT) and Audio to perform **Bidirectional Multi-Head Cross-Attention** and **Audio-Guided Gated Multimodal Fusion (GMU)**. 
- **Multi-task Learning & Knowledge Distillation**: The Teacher transfers knowledge to the Student via:
  - **KL Divergence (KD Loss)**: Aligning student and teacher emotion logits.
  - **Hallucination Loss (Cosine Similarity)**: Encouraging the student to hallucinate representations highly similar to the teacher's.
  - **CTC Loss**: For auxiliary ASR training.
- **Standard Benchmarking**: Built-in support for IEMOCAP evaluation (LOSO-5-fold cross-validation on 4 emotion classes: neutral, happy, angry, sad) and imbalanced data handling via Class Weights.

---

## 📂 Directory Structure

```text
ViSER/
├── config/
│   └── config.yaml          # Hyperparameters configuration file
├── vi_ser/                  # Core package
│   ├── data_loader/         # DataLoader and Dataset processing
│   ├── encoders/            # Feature extraction modules (Wav2Vec2, BERT)
│   ├── fusion/              # Logit-Guided Hallucination, CrossModal, GMU, Classifiers
│   ├── config.py            # Default configuration dataclass
│   ├── loss.py              # Combined multi-objective loss function
│   └── model.py             # SERModel architecture assembly
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
└── requirements.txt         # Dependencies list
```

---

## ⚙️ Installation

Requirements: Python 3.8+ and CUDA (GPU) support.

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏃 Usage

### 1. Data Preparation
Ensure you have the CSV files containing the dataset metadata. The paths to these files are configured in `config/config.yaml`. The CSV format should include columns for: audio file path, emotion label, and transcription text (for Teacher training).

### 2. Training
Run the `train.py` script to start training. The model will automatically load configurations from `config/config.yaml`.

```bash
python train.py --config config/config.yaml
```
*Note: The top best checkpoints will be automatically saved in the `checkpoints/` directory.*

You can also override hyperparameters directly from the command line:
```bash
python train.py --override training.batch_size=8 loss.alpha_kd=0.5
```

### 3. Evaluation
During evaluation, the model runs in **End-to-End Audio-Only (Student)** mode. No text transcriptions are required.

```bash
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_X_acc_Y.pt
```

---

## 🧩 Multi-Objective Loss Mechanism
The total loss of ViSER is a combination of 6 different objectives to optimize the Dual-Branch setup and Knowledge Distillation:

`L_total = α_s_emo * L_emotion_student + α_t_emo * L_emotion_teacher + α_ctc * L_ctc + α_kd * L_kd + α_distill * L_distill + λ_hallucination * L_hallucination`

You can adjust these weights inside the `config/config.yaml` file to balance the multi-task learning.

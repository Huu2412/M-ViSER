# M-ViSER (AURORA): Multimodal Speech Emotion Recognition

M-ViSER is a state-of-the-art multimodal speech emotion recognition (SER) system featuring the robust AURORA fusion architecture. It natively supports both standard English benchmarking on IEMOCAP and is extensible to other languages like Vietnamese.

## 🚀 Key Features

- **Acoustic Backbone**: Utilizes Wav2Vec2 (e.g., `facebook/wav2vec2-base-960h`) for robust acoustic feature extraction and CTC representation.
- **Text Backbone**: Employs BERT (e.g., `bert-base-uncased`) for deep semantic text processing.
- **ASR Teacher**: Uses Whisper models as a "Teacher" to provide high-quality transcriptions and distill robust linguistic knowledge.
- **AURORA Fusion Architecture**: Integrates a **Bidirectional Multi-Head Cross-Attention** module for cross-modal interaction, a **Repair MLP** (to correct ASR spelling errors), and an **Audio-Guided Gated Multimodal Unit (GMU)** with an Uncertainty Gate, allowing the model to dynamically assess the reliability of the generated text.
- **Multi-task Learning & Knowledge Distillation**: The model is trained using a Dual-Branch approach where the Teacher (Clean Text) distills knowledge to the Student (ASR Text) via KL divergence and Representation Alignment (MSE). It also performs Connectionist Temporal Classification (CTC) as an auxiliary ASR task.
- **Standard Benchmarking**: Built-in support for IEMOCAP standard evaluation (LOSO-5-fold cross-validation on 4 emotion classes: neutral, happy, angry, sad) and imbalanced data handling via Class Weights.

---

## 📂 Directory Structure

The project is organized following standard Python Package structure conventions:

```text
ViSER/
├── config/
│   └── config.yaml          # Hyperparameters configuration file
├── scripts/
│   └── precache_teacher.py  # Script to pre-cache PhoWhisper ASR texts
├── vi_ser/                  # Core package containing the algorithm
│   ├── data_loader/         # DataLoader and Dataset processing
│   ├── encoders/            # Feature extraction modules (Acoustic, Text)
│   ├── fusion/              # Fusion modules (CrossModal, Repair Gate, GMU, Classifiers)
│   ├── config.py            # Default configuration dataclass
│   ├── loss.py              # Combined loss function
│   └── model.py             # ViSERModel architecture assembly
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
└── requirements.txt         # Dependencies list
```

---

## ⚙️ Installation

Requirements: Python 3.8+ and CUDA (GPU) support for optimal training speed.

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏃 Usage

### 1. Data Preparation
Ensure you have the CSV files containing the dataset metadata. The paths to these files are configured in `config/config.yaml` (e.g., `train_csv`, `val_csv`, `test_csv`). The CSV format should include columns for: audio file path, emotion label, and regional label.

**Recommendation:** Before training, run the precaching script to extract clean transcripts using PhoWhisper. This speeds up the main training loop significantly:
```bash
python scripts/precache_teacher.py --config config/config.yaml --batch_size 8
```

### 2. Training
Run the `train.py` script to start training. The model will automatically load configurations from `config/config.yaml`.

```bash
python train.py --config config/config.yaml
```
*Note: The top 3 best checkpoints will be automatically saved in the `checkpoints/` directory.*

You can also override hyperparameters directly from the command line:
```bash
python train.py --override training.batch_size=8 loss.alpha_regional=0.2
```

### 3. Evaluation
To evaluate the model's accuracy on the Test set:

```bash
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_X_acc_Y.pt
```

---

## 🧩 Loss Mechanism & Hyperparameters
The total loss of ViSER is a combination of 5 different objectives to optimize the Dual-Branch setup and Knowledge Distillation:
`L_total = α_s_emo * L_emotion_student + α_t_emo * L_emotion_teacher + α_ctc * L_ctc + α_kd * L_kd + α_distill * L_distill`

You can easily adjust these `α` (alpha) weights, as well as define Class Weights (`emotion_class_weights`) inside the `config/config.yaml` file to handle imbalanced data.

"""
vi_ser/config.py
Centralized configuration dataclass for the SER model.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class ViSERConfig:
    experiment_name: str = "ser_mtl_aurora"

    # ── Backbone (Acoustic Encoder) ──────────────────────────────────────────
    acoustic_model_name: str = "facebook/wav2vec2-base-960h"
    acoustic_hidden_size: int = 768
    freeze_feature_extractor: bool = True
    freeze_acoustic_encoder: bool = False

    # ── Text Encoder (BERT) ──────────────────────────────────────────────────
    text_model_name: str = "bert-base-uncased"
    text_hidden_size: int = 768
    freeze_text_encoder: bool = False

    # ── Fusion Dimensions ────────────────────────────────────────────────────
    fusion_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.3
    repair_hidden_dim: int = 256
    delta_scale: float = 0.3
    uncertainty_alpha_min: float = 0.05
    uncertainty_alpha_max: float = 0.95
    repair_use_alpha: bool = False  # Ablation flag: if True, z_repaired = z_asr + alpha * delta

    # ── Classifier Heads ─────────────────────────────────────────────────────
    num_emotion_classes: int = 4
    emotion_class_weights: Optional[list] = None
    classifier_hidden_dim: int = 256
    vocab_size: int = 32  # will be overridden from CTC tokenizer
    pad_token_id: int = 0 # blank token for CTC loss

    # ── Loss Weights ─────────────────────────────────────────────────────────
    alpha_student_emotion: float = 1.0
    alpha_teacher_emotion: float = 0.5
    alpha_ctc: float = 0.2
    alpha_kd: float = 0.5
    alpha_distill: float = 0.0
    kd_temperature: float = 2.0
    label_smoothing: float = 0.05

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs: int = 50
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # LRs
    learning_rate: float = 2e-4
    backbone_lr: float = 1e-5
    head_lr: float = 5e-4

    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"
    scheduler_min_lr: float = 1e-6
    scheduler_warmup_epochs: int = 3

    checkpoint_metric: str = "macro_f1"

    max_audio_length_sec: float = 10.0
    sampling_rate: int = 16000
    ctc_zero_infinity: bool = True
    seed: int = 42
    num_workers: int = 0

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "checkpoints"
    cache_dir: Optional[str] = "cache_ser"
    log_dir: str = "logs_ser"

    # ── Labels ────────────────────────────────────────────────────────────────
    emotion_label_map: dict = field(default_factory=lambda: {
        "neu": 0, "hap": 1, "ang": 2, "sad": 3
    })

    # ── Dataset ────────────────────────────────────────────────────────────────
    hf_dataset: Optional[str] = None
    current_fold: int = 1

    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    test_csv: str = "data/test.csv"
    speech_col: str = "file"
    text_col: str = "text"
    emotion_col: str = "emotion"

    augment_prob: float = 0.0
    augment_pitch_weight: float = 0.4
    augment_pitch_steps: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    augment_noise_weight: float = 0.4
    augment_noise_range: List[float] = field(default_factory=lambda: [0.001, 0.015])
    augment_time_weight: float = 0.2
    augment_time_range: List[float] = field(default_factory=lambda: [0.9, 1.1])

    # ── Misc ──────────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "ser_mtl_aurora"
    log_steps: int = 50

"""
vi_ser/config.py
Centralized configuration dataclass for the ViSER model.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class ViSERConfig:
    experiment_name: str = "viser_optimized"

    # ── Backbone (Acoustic Encoder) ──────────────────────────────────────────
    acoustic_model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    acoustic_hidden_size: int = 768
    freeze_feature_extractor: bool = True
    freeze_acoustic_encoder: bool = False

    # ── ASR Teacher (PhoWhisper) ─────────────────────────────────────────────
    phowhisper_model_name: str = "vinai/PhoWhisper-medium"
    freeze_teacher: bool = True

    # ── Text Encoder (PhoBERT) ───────────────────────────────────────────────
    phobert_model_name: str = "vinai/phobert-base"
    text_hidden_size: int = 768
    freeze_phobert: bool = False

    # ── Fusion Dimensions ────────────────────────────────────────────────────
    fusion_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.3
    repair_hidden_dim: int = 256
    delta_scale: float = 0.3
    uncertainty_alpha_min: float = 0.05
    uncertainty_alpha_max: float = 0.95

    # ── Classifier Heads ─────────────────────────────────────────────────────
    num_emotion_classes: int = 4
    num_regional_classes: int = 3
    emotion_class_weights: Optional[list] = None
    regional_class_weights: Optional[list] = None
    classifier_hidden_dim: int = 256
    vocab_size: int = 64

    # ── Loss Weights ─────────────────────────────────────────────────────────
    alpha_student_emotion: float = 0.5
    alpha_teacher_emotion: float = 0.5
    alpha_ctc: float = 0.1
    alpha_kd: float = 0.0
    alpha_distill: float = 0.0
    alpha_regional: float = 0.2
    kd_temperature: float = 2.0
    label_smoothing: float = 0.1

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs: int = 50
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # LRs
    learning_rate: float = 2e-4
    backbone_lr: float = 1e-4
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
    num_workers: int = 4

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "checkpoints"
    cache_dir: Optional[str] = "cache_vi_ser"
    log_dir: str = "logs_vi_ser"

    # ── Labels ────────────────────────────────────────────────────────────────
    emotion_label_map: dict = field(default_factory=lambda: {
        "e0": 0, "e1": 1, "e2": 2, "e3": 3
    })
    regional_label_map: dict = field(default_factory=lambda: {
        "north": 0, "central": 1, "south": 2
    })

    # ── Dataset & Augmentation ────────────────────────────────────────────────
    hf_dataset: Optional[str] = None
    current_fold: int = 1
    
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    test_csv: str = "data/test.csv"
    speech_col: str = "file"
    text_col: str = "text"
    emotion_col: str = "emotion"
    regional_col: str = "regional"
    
    augment_prob: float = 0.7
    augment_pitch_weight: float = 0.4
    augment_pitch_steps: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    augment_noise_weight: float = 0.4
    augment_noise_range: List[float] = field(default_factory=lambda: [0.001, 0.015])
    augment_time_weight: float = 0.2
    augment_time_range: List[float] = field(default_factory=lambda: [0.9, 1.1])

    # ── Misc ──────────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "viser"
    log_steps: int = 50

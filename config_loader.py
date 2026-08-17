"""
config_loader.py

Loads ViSERConfig from a YAML file.
"""

import yaml
from vi_ser.config import ViSERConfig


def load_config(yaml_path: str) -> ViSERConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Root
    exp_name = raw.get("experiment_name", "ser_mtl_aurora")

    p   = raw.get("paths", {})
    dat = raw.get("dataset", {})
    mod = raw.get("model", {})
    frz = raw.get("freeze", {})
    lss = raw.get("loss", {})
    tr  = raw.get("training", {})
    log = raw.get("logging", {})

    # Augmentations
    aug = dat.get("augmentations", {}).get("waveform_augment", {})
    pitch   = aug.get("pitch_shift", {})
    noise   = aug.get("noise_injection", {})
    time_sh = aug.get("time_shift", {})

    # Scheduler
    sched = tr.get("scheduler", {})

    config = ViSERConfig(
        experiment_name = exp_name,

        # paths
        train_csv  = p.get("train_csv",  "data/train.csv"),
        val_csv    = p.get("val_csv",    "data/val.csv"),
        test_csv   = p.get("test_csv",   "data/test.csv"),
        output_dir = p.get("output_dir", "checkpoints"),
        cache_dir  = p.get("cache_dir",  "cache_ser"),
        log_dir    = p.get("log_dir",    "logs_ser"),

        # dataset
        hf_dataset        = dat.get("hf_dataset",   None),
        current_fold      = dat.get("current_fold", 1),
        speech_col        = dat.get("speech_col",   "file"),
        text_col          = dat.get("text_col",     "text"),
        emotion_col       = dat.get("emotion_col",  "emotion"),
        emotion_label_map = dat.get("emotion_label_map", {"neu":0,"hap":1,"ang":2,"sad":3}),

        augment_prob         = aug.get("prob", 0.0),
        augment_pitch_weight = pitch.get("weight", 0.4),
        augment_pitch_steps  = pitch.get("n_steps_range", [-1.0, 1.0]),
        augment_noise_weight = noise.get("weight", 0.4),
        augment_noise_range  = noise.get("noise_factor_range", [0.001, 0.015]),
        augment_time_weight  = time_sh.get("weight", 0.2),
        augment_time_range   = time_sh.get("range", [0.9, 1.1]),

        # models
        acoustic_model_name  = mod.get("acoustic",   "facebook/wav2vec2-base-960h"),
        text_model_name      = mod.get("text_model", "bert-base-uncased"),
        fusion_dim            = mod.get("fusion_dim",             512),
        repair_hidden_dim     = mod.get("repair_hidden_dim",      256),
        classifier_hidden_dim = mod.get("classifier_hidden_dim",  256),
        num_emotion_classes   = mod.get("num_emotion_classes",    4),
        num_heads             = mod.get("num_heads",              8),
        dropout               = mod.get("dropout",                0.3),
        delta_scale           = mod.get("delta_scale",            0.3),
        uncertainty_alpha_min = mod.get("uncertainty_alpha_min",  0.05),
        uncertainty_alpha_max = mod.get("uncertainty_alpha_max",  0.95),

        # freeze
        freeze_feature_extractor = frz.get("feature_extractor", True),
        freeze_acoustic_encoder  = frz.get("acoustic_encoder",  False),
        freeze_text_encoder      = frz.get("text_encoder",      False),

        # loss
        alpha_student_emotion = lss.get("alpha_student_emotion", 1.0),
        alpha_teacher_emotion = lss.get("alpha_teacher_emotion", 0.5),
        alpha_ctc             = lss.get("alpha_ctc",      0.2),
        alpha_kd              = lss.get("alpha_kd",       0.5),
        alpha_distill         = lss.get("alpha_distill",  0.0),
        kd_temperature        = lss.get("kd_temperature", 2.0),

        # training
        num_epochs                  = tr.get("epochs",                      50),
        batch_size                  = tr.get("batch_size",                  8),
        gradient_accumulation_steps = tr.get("gradient_accumulation_steps", 4),
        learning_rate               = tr.get("lr",                          2e-4),
        backbone_lr                 = tr.get("backbone_lr",                 1e-5),
        head_lr                     = tr.get("head_lr",                     5e-4),
        weight_decay                = tr.get("weight_decay",                0.01),
        grad_clip_norm              = tr.get("grad_clip_norm",              1.0),
        label_smoothing             = tr.get("label_smoothing",             0.05),
        checkpoint_metric           = tr.get("checkpoint_metric",           "macro_f1"),
        max_audio_length_sec        = tr.get("max_audio_length_sec",        10.0),
        sampling_rate               = tr.get("sampling_rate",               16000),
        ctc_zero_infinity           = tr.get("ctc_zero_infinity",           True),
        seed                        = tr.get("seed",                        42),
        num_workers                 = tr.get("num_workers",                 0),

        # scheduler
        scheduler_type          = sched.get("type", "cosine"),
        scheduler_min_lr        = float(sched.get("min_lr", 1e-6)),
        scheduler_warmup_epochs = sched.get("warmup_epochs", 3),

        # logging
        use_wandb     = log.get("use_wandb",     False),
        wandb_project = log.get("wandb_project", "ser_mtl_aurora"),
        log_steps     = log.get("log_steps",     50),
    )
    return config

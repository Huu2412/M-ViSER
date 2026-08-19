"""
train.py  —  Training script for ViSER model.

Usage:
    # Dùng config file (khuyến nghị)
    python train.py --config config.yaml

    # Override một vài tham số từ CLI
    python train.py --config config.yaml --override training.learning_rate=1e-4 loss.alpha_kd=0.8
"""

import os
import sys
import argparse
import logging
import random
import json
import warnings

# Giảm thiểu các log thừa từ thư viện
import transformers
import datasets
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vi_ser.data_loader.iemocap import build_dataloaders
from config_loader import load_config
from vi_ser.factory import (
    create_model,
    create_loss,
    create_optimizer,
    create_scheduler,
    create_acoustic_feature_extractor,
    create_ctc_tokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def evaluate(model, val_loader, loss_fn, device, config, ctc_tokenizer):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    emotion_correct = 0
    total = 0
    
    all_emotion_preds = []
    all_emotion_labels = []

    start_eval_time = time.time()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval", disable=True):
            input_values  = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            emotion_labels  = batch["emotion_labels"].to(device)
            ctc_labels      = batch.get("ctc_labels")
            if ctc_labels is not None:
                ctc_labels = ctc_labels.to(device)
            student_texts   = batch["student_texts"]
            teacher_texts   = batch["teacher_texts"]

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                student_texts=student_texts,
                teacher_texts=None,
                processor=ctc_tokenizer,
                training_mode=False,
            )

            loss, loss_dict = loss_fn(
                logits_emotion_student=outputs["logits_emotion_student"],
                logits_ctc=outputs["logits_ctc"],
                z_fused=outputs["z_fused"],
                emotion_labels=emotion_labels,
                ctc_labels=ctc_labels,
                input_values=input_values,
                attention_mask=attention_mask,
                logits_emotion_teacher=outputs.get("logits_emotion_teacher"),
                z_teacher_rep=outputs.get("z_teacher_rep"),
                acoustic_encoder=outputs["acoustic_encoder"],
            )

            B = input_values.size(0)
            total_loss += loss.item() * B
            
            emo_preds = outputs["logits_emotion_student"].argmax(-1)
            emotion_correct  += int((emo_preds == emotion_labels).sum())
            
            all_emotion_preds.extend(emo_preds.cpu().tolist())
            all_emotion_labels.extend(emotion_labels.cpu().tolist())
            total += B

    from sklearn.metrics import f1_score, accuracy_score, recall_score
    macro_f1 = f1_score(all_emotion_labels, all_emotion_preds, average="macro")
    wa = accuracy_score(all_emotion_labels, all_emotion_preds)
    ua = recall_score(all_emotion_labels, all_emotion_preds, average="macro", zero_division=0)

    end_eval_time = time.time()
    inf_time_ms = ((end_eval_time - start_eval_time) / max(total, 1)) * 1000

    return {
        "val_loss":            total_loss / total,
        "emotion_acc":         emotion_correct / total,   # WA = emotion_acc
        "wa":                  wa,
        "ua":                  ua,
        "macro_f1":            macro_f1,
        "inf_time_ms":         inf_time_ms,
    }


def train(config):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(
            {k: str(v) for k, v in vars(config).items()},
            f, indent=2, ensure_ascii=False
        )

    # ── Load feature extractor ───────────────────────────────────────────────
    logger.info("Loading Wav2Vec2 feature extractor...")
    feature_extractor = create_acoustic_feature_extractor(config)

    # ── Load CTC tokenizer ───────────────────────────────────────────────────
    logger.info("Loading CTC tokenizer...")
    ctc_tokenizer = create_ctc_tokenizer(config)
    config.vocab_size = len(ctc_tokenizer)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    # Teacher text (clean GT transcripts) is read from the dataset CSV text column.
    # No external ASR teacher model needed.
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        config, feature_extractor, ctc_tokenizer,
    )

    # ── Initialize Model ─────────────────────────────────────────────────────
    logger.info("Initializing ViSER model...")
    model = create_model(config, device)
    # CRITICAL: Force float32. Wav2Vec2 attention overflows in float16 on GPU
    # causing NaN in hidden_states for long/loud audio sequences.
    model = model.float()

    param_counts = model.count_parameters()
    logger.info("Trainable parameters per module:")
    for name, count in param_counts.items():
        logger.info(f"  {name}: {count:,}")

    # ── Loss & Optimizer ─────────────────────────────────────────────────────
    loss_fn = create_loss(config)
    optimizer = create_optimizer(model, config)
    
    steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

    # ── Compute FLOPs (using thop if available) ──────────────────────────────
    flops_str = "N/A"
    try:
        import thop
        logger.info("Computing FLOPs with thop...")
        # Create a dummy input (1 second audio, no text since ViSEC is audio-only)
        dummy_audio = torch.randn(1, 16000, device=device)
        dummy_mask = torch.ones(1, 16000, dtype=torch.long, device=device)
        # Using a wrapper to match forward signature for thop
        class ModelWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, a, m): return self.m(input_values=a, attention_mask=m, student_texts=[""], teacher_texts=[""], training_mode=False)
        macs, _ = thop.profile(ModelWrapper(model), inputs=(dummy_audio, dummy_mask), verbose=False)
        flops_str = f"{macs * 2 / 1e9:.2f}G"  # 1 MAC = 2 FLOPs
        logger.info(f"FLOPs (1s audio): {flops_str}")
    except Exception as e:
        logger.info("Install 'thop' (pip install thop) to calculate FLOPs dynamically.")

    # ── Training Loop ─────────────────────────────────────────────────────────
    top_k_checkpoints = []  # List of tuples: (emotion_acc, save_path)
    max_checkpoints = 3
    
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50 + "\n")

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_losses = {
            "l_total": 0.0, "l_emotion_student": 0.0, "l_emotion_teacher": 0.0, "l_emotion": 0.0, "l_ctc": 0.0,
            "l_kd": 0.0, "l_distill": 0.0,
        }
        n_batches = 0
        train_emotion_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", disable=True)
        for step, batch in enumerate(pbar):
            input_values = batch["input_values"].to(device).float()  # force float32
            # CRITICAL: Add tiny noise to break exact 0.0 padding. Wav2Vec2 feature extractor
            # produces NaN internally if given perfectly silent (0.0) padding frames.
            input_values = input_values + torch.randn_like(input_values) * 1e-6
            
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            emotion_labels  = batch["emotion_labels"].to(device)
            ctc_labels      = batch.get("ctc_labels")
            if ctc_labels is not None:
                ctc_labels = ctc_labels.to(device)
            student_texts   = batch["student_texts"]
            teacher_texts   = batch["teacher_texts"]

            # ── Forward pass ──────────────────────────────────────────────
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                student_texts=student_texts,
                teacher_texts=teacher_texts,
                processor=ctc_tokenizer,
                training_mode=True,
            )

            # ── Compute loss ──────────────────────────────────────────────
            loss, loss_dict = loss_fn(
                logits_emotion_student=outputs["logits_emotion_student"],
                logits_ctc=outputs["logits_ctc"],
                z_fused=outputs["z_fused"],
                emotion_labels=emotion_labels,
                ctc_labels=ctc_labels,
                input_values=input_values,
                attention_mask=attention_mask,
                logits_emotion_teacher=outputs.get("logits_emotion_teacher"),
                z_teacher_rep=outputs.get("z_teacher_rep"),
                acoustic_encoder=outputs["acoustic_encoder"],
            )

            # ── Safe Guard: Check for NaN/Inf loss ───────────────────────────
            if not torch.isfinite(loss):
                logger.warning(f"NaN/Inf loss detected at step {step} (L_emo={loss_dict.get('l_emotion_student')}, L_ctc={loss_dict.get('l_ctc')}). Skipping batch!")
                optimizer.zero_grad()
                continue

            # ── Backward & gradient accumulation ─────────────────────────
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                # CRITICAL: Prevent NaN gradients from corrupting optimizer state
                # Tối ưu hoá: Sử dụng trực tiếp grad_norm trả về từ clip_grad_norm_ thay vì loop tay
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(config, "grad_clip_norm", 1.0))
                
                if not torch.isfinite(grad_norm):
                    logger.warning(f"NaN/Inf gradients detected at step {step} (norm={grad_norm})! Skipping optimizer step to save weights.")
                else:
                    optimizer.step()
                    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                        
                optimizer.zero_grad()

            # Track losses & accuracy
            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0.0)
            n_batches += 1
            
            B = input_values.size(0)
            train_total += B
            with torch.no_grad():
                emo_preds = outputs["logits_emotion_student"].argmax(-1)
                train_emotion_correct += int((emo_preds == emotion_labels).sum())

            pbar.set_postfix({
                "loss": f"{loss_dict['l_total']:.4f}",
                "emo":  f"{loss_dict['l_emotion']:.4f}",
                "kd":   f"{loss_dict['l_kd']:.4f}",
            })

        # ── Epoch summary ─────────────────────────────────────────────────
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        
        train_emo_acc = train_emotion_correct / max(train_total, 1) * 100
        
        # ── Validation ────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, loss_fn, device, config, ctc_tokenizer)
        
        epoch_time = time.time() - epoch_start_time
        
        # In ra màn hình giống định dạng MaxMViT-MLP-SER
        # VD: Epoch 01 | Train [L:2.4051 L_emo:1.2051 L_ctc:0.1234 A:54.3%] | Val [L:1.2141 A:65.4% mF1:64.2%] | Time: 120.5s
        log_line = (
            f"Epoch {epoch+1:02d} | "
            f"Train [L:{epoch_losses['l_total']:.4f} L_emo:{epoch_losses['l_emotion_student']:.4f} L_ctc:{epoch_losses['l_ctc']:.4f} A:{train_emo_acc:.1f}%] | "
            f"Val [L:{val_metrics['val_loss']:.4f} A:{val_metrics.get('wa', 0)*100:.1f}% mF1:{val_metrics['macro_f1']*100:.1f}%] | "
            f"Time: {epoch_time:.1f}s"
        )
        logger.info(log_line)
        print(log_line)
        sys.stdout.flush()

        # Scheduler step (only for Plateau, step-based schedulers are updated per batch)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics.get(config.checkpoint_metric, val_metrics["macro_f1"]))

        # ── Save top-k models ───────────────────────────────────────────────
        current_metric = val_metrics.get(config.checkpoint_metric, val_metrics["macro_f1"])
        save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}_{config.checkpoint_metric}_{current_metric:.4f}.pt")
        
        top_k_checkpoints.append((current_metric, save_path))
        top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        if (current_metric, save_path) in top_k_checkpoints[:max_checkpoints]:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": vars(config),
            }, save_path)
            
            is_best = (current_metric, save_path) == top_k_checkpoints[0]
            if is_best:
                best_path = os.path.join(config.output_dir, "best_model.pt")
                import shutil
                shutil.copyfile(save_path, best_path)
                logger.info(f"  🌟 New best model! Saved to {best_path}")
            else:
                logger.info(f"  ✓ Saved checkpoint ({config.checkpoint_metric}={current_metric:.4f}) -> {save_path}")
            
            if len(top_k_checkpoints) > max_checkpoints:
                worst_acc, worst_path = top_k_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    logger.info(f"  - Removed old checkpoint: {worst_path}")
        else:
            top_k_checkpoints.pop()

    best_metric = top_k_checkpoints[0][0] if top_k_checkpoints else 0.0
    logger.info(f"Training complete. Best {config.checkpoint_metric}: {best_metric:.4f}")


def _apply_overrides(config, overrides: list):
    """
    Apply CLI overrides to config.
    Format: "section.key=value"  e.g. "training.learning_rate=1e-4"
    """
    SECTION_FIELD_MAP = {
        # training.*
        "training.num_epochs":                  ("num_epochs",                  int),
        "training.batch_size":                  ("batch_size",                  int),
        "training.learning_rate":               ("learning_rate",               float),
        "training.gradient_accumulation_steps": ("gradient_accumulation_steps", int),
        "training.seed":                        ("seed",                        int),
        "training.weight_decay":                ("weight_decay",                float),
        # loss.*
        "loss.alpha_student_emotion": ("alpha_student_emotion", float),
        "loss.alpha_teacher_emotion": ("alpha_teacher_emotion", float),
        "loss.alpha_ctc":       ("alpha_ctc",       float),
        "loss.alpha_kd":        ("alpha_kd",         float),
        "loss.alpha_distill":   ("alpha_distill",   float),
        "loss.kd_temperature":  ("kd_temperature",  float),
        # architecture.*
        "architecture.fusion_dim":            ("fusion_dim",            int),
        "architecture.dropout":               ("dropout",               float),
        "architecture.num_emotion_classes":   ("num_emotion_classes",   int),
        # paths.*
        "paths.train_csv":   ("train_csv",   str),
        "paths.val_csv":     ("val_csv",     str),
        "paths.output_dir":  ("output_dir",  str),
        "paths.cache_dir":   ("cache_dir",   str),
        "paths.log_dir":     ("log_dir",     str),
        # dataset.*
        "dataset.current_fold": ("current_fold", int),
        "dataset.hf_dataset":   ("hf_dataset",   str),
    }
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Invalid override (missing '='): {override}")
            continue
        key, value = override.split("=", 1)
        if key in SECTION_FIELD_MAP:
            field_name, cast_fn = SECTION_FIELD_MAP[key]
            setattr(config, field_name, cast_fn(value))
            logger.info(f"Override: {field_name} = {cast_fn(value)}")
        else:
            logger.warning(f"Unknown override key: {key}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViSER model")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="section.key=value",
        help="Override config values, e.g. --override training.learning_rate=1e-4 loss.alpha_kd=0.8"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded config from: {args.config}")

    if args.override:
        config = _apply_overrides(config, args.override)

    train(config)

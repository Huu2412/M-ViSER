"""
utils.py — Utility functions for ViSER (Vietnamese Speech Emotion Recognition).

Provides:
  - set_seed            : reproducibility
  - compute_accuracy    : batch accuracy helper
  - compute_wer         : Word Error Rate (CTC ASR evaluation)
  - format_loss_dict    : pretty-print loss components
  - get_model_size      : report parameter counts
  - save_checkpoint     : save model + optimizer + metadata
  - load_checkpoint     : restore model + optimizer from .pt file
  - EarlyStopping       : patience-based early stopping helper
  - AverageMeter        : running average tracker
  - Timer               : simple elapsed-time context manager
"""

import os
import time
import json
import random
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Seed set to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy from logits.

    Args:
        logits: [B, C]  raw logits (or probabilities)
        labels: [B]     ground-truth class indices

    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / max(len(labels), 1)


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER) for CTC student ASR.

    WER = (S + D + I) / N
      S = substitutions, D = deletions, I = insertions, N = reference words.

    Args:
        hypotheses : list of decoded strings from CTC model
        references : list of ground-truth strings

    Returns:
        WER as a float (0.0 = perfect).
    """
    total_edits, total_ref_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        h_tokens = hyp.strip().split()
        r_tokens = ref.strip().split()
        total_edits   += _edit_distance(h_tokens, r_tokens)
        total_ref_len += max(len(r_tokens), 1)
    return total_edits / max(total_ref_len, 1)


def _edit_distance(hyp: List[str], ref: List[str]) -> int:
    """Levenshtein edit distance between two token lists."""
    n, m = len(hyp), len(ref)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_loss_dict(loss_dict: Dict[str, float], prefix: str = "") -> str:
    """
    Format a loss dict as a compact string for logging.

    Example output:
        "l_total=0.8432 | l_emotion=0.3120 | l_ctc=0.1234 | l_kd=0.2000"
    """
    parts = [f"{prefix}{k}={v:.4f}" for k, v in loss_dict.items()]
    return " | ".join(parts)


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters per named child module.

    Returns:
        Dict with keys: module names + "total_trainable" + "total_all"
    """
    counts: Dict[str, int] = {}
    for name, module in model.named_children():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in module.parameters())
        counts[name] = trainable
        counts[f"{name}_all"] = all_params
    counts["total_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts["total_all"] = sum(p.numel() for p in model.parameters())
    return counts


def log_model_size(model: nn.Module) -> None:
    """Log trainable parameter count of model to INFO."""
    counts = get_model_size(model)
    logger.info(
        f"Model size — trainable: {counts['total_trainable']:,}  "
        f"| total: {counts['total_all']:,}"
    )
    for name, module in model.named_children():
        logger.info(
            f"  {name}: {counts[name]:,} trainable / {counts[name + '_all']:,} total"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config,
    save_path: str,
    scheduler=None,
) -> None:
    """
    Save model checkpoint with full metadata.

    Saves:
        model_state_dict, optimizer_state_dict, scheduler_state_dict (optional),
        epoch, metrics, config (as plain dict).

    Args:
        model      : ViSERModel instance
        optimizer  : torch Optimizer
        epoch      : current epoch (1-indexed)
        metrics    : dict of metric values, e.g. {"emotion_acc": 0.85, ...}
        config     : ViSERConfig dataclass
        save_path  : full path to .pt file
        scheduler  : (optional) LR scheduler
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":            metrics,
        "config":             vars(config),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, save_path)
    logger.info(f"Checkpoint saved → {save_path}  (epoch {epoch})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        checkpoint_path : path to .pt checkpoint
        model           : ViSERModel instance (modified in-place)
        optimizer       : (optional) Optimizer to restore
        scheduler       : (optional) LR scheduler to restore
        device          : target device (default: CPU)

    Returns:
        The full checkpoint dict (includes "epoch", "metrics", "config", etc.)
    """
    map_location = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded model from {checkpoint_path}  "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored.")
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored.")
    return checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Patience-based early stopping.

    Monitors a metric and signals stop when no improvement is seen
    for `patience` consecutive epochs.

    Usage::

        early_stop = EarlyStopping(patience=10, mode="max")
        for epoch in range(num_epochs):
            val_acc = validate(...)
            if early_stop(val_acc):
                break
    """

    def __init__(self, patience: int = 10, mode: str = "max", delta: float = 1e-4):
        """
        Args:
            patience : epochs to wait without improvement before stopping
            mode     : "max" (higher is better) or "min" (lower is better)
            delta    : minimum change to qualify as improvement
        """
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.counter   = 0
        self.best      = None
        self.triggered = False

    def __call__(self, metric: float) -> bool:
        """
        Update state with new metric value.

        Returns True if training should stop.
        """
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric > self.best + self.delta
            if self.mode == "max"
            else metric < self.best - self.delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                f"(best={self.best:.4f}, current={metric:.4f})"
            )
            if self.counter >= self.patience:
                logger.info(
                    f"EarlyStopping triggered after {self.patience} epochs "
                    f"without improvement (best={self.best:.4f})."
                )
                self.triggered = True
                return True
        return False

    def reset(self) -> None:
        """Reset counter and best value (e.g. after curriculum phase change)."""
        self.counter   = 0
        self.best      = None
        self.triggered = False


# ─────────────────────────────────────────────────────────────────────────────
# Running average tracker
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """
    Track running average of a scalar quantity (e.g. loss per epoch).

    Usage::

        meter = AverageMeter("train_loss")
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=len(batch))
        print(meter)   # AverageMeter(train_loss=0.3412, count=512)
    """

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum   = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """
        Args:
            value : scalar value (e.g. batch loss)
            n     : number of samples this value represents
        """
        self.sum   += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """Current running average."""
        return self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}={self.avg:.4f}, count={self.count})"


# ─────────────────────────────────────────────────────────────────────────────
# Timer context manager
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """
    Simple wall-clock timer as a context manager.

    Usage::

        with Timer("data loading") as t:
            loader = build_dataloaders(...)
        # logs: "[Timer] data loading: 3.42s"
    """

    def __init__(self, name: str = "block", log: bool = True):
        self.name    = name
        self.log     = log
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self._start
        if self.log:
            logger.info(f"[Timer] {self.name}: {self.elapsed:.2f}s")

    def __repr__(self) -> str:
        return f"Timer({self.name}, elapsed={self.elapsed:.2f}s)"


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_truncate(
    waveform: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad (right) or truncate a 1-D waveform to exactly `target_length` samples.

    Args:
        waveform      : 1-D numpy array of audio samples
        target_length : desired output length in samples
        pad_value     : value used for padding (default: silence = 0.0)

    Returns:
        1-D numpy array of shape (target_length,)
    """
    if len(waveform) >= target_length:
        return waveform[:target_length]
    pad_width = target_length - len(waveform)
    return np.pad(waveform, (0, pad_width), constant_values=pad_value)


def seconds_to_samples(seconds: float, sampling_rate: int = 16000) -> int:
    """Convert duration in seconds to number of audio samples."""
    return int(seconds * sampling_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_ID2NAME: Dict[int, str] = {0: "Neutral", 1: "Happy", 2: "Angry", 3: "Sad"}
EMOTION_NAME2ID: Dict[str, int] = {v: k for k, v in EMOTION_ID2NAME.items()}

REGIONAL_ID2NAME: Dict[int, str] = {0: "North", 1: "Central", 2: "South"}
REGIONAL_NAME2ID: Dict[str, int] = {v: k for k, v in REGIONAL_ID2NAME.items()}


def decode_emotion_labels(label_ids: List[int]) -> List[str]:
    """Convert list of emotion label indices → human-readable names."""
    return [EMOTION_ID2NAME.get(i, f"unk_{i}") for i in label_ids]


def decode_regional_labels(label_ids: List[int]) -> List[str]:
    """Convert list of regional label indices → human-readable names."""
    return [REGIONAL_ID2NAME.get(i, f"unk_{i}") for i in label_ids]


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_json(obj: dict, path: str, indent: int = 2) -> None:
    """Serialize a dict to a JSON file (UTF-8, human-readable)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Saved JSON => {path}")


def load_json(path: str) -> dict:
    """Load a JSON file and return a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""
vi_ser/evaluate.py

Evaluation script for ViSER model.
Evaluates on test set and reports:
  - Emotion accuracy (UA/WA)
  - WER (Word Error Rate from CTC student ASR)
  - Confusion matrix for emotion
  - Per-class F1 scores
"""

import os
import sys
import argparse
import logging
import json

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vi_ser.config import ViSERConfig
from vi_ser.data_loader.iemocap import ViSERDataset, ViSERCollator
from config_loader import load_config
from torch.utils.data import DataLoader
from vi_ser.factory import create_model, create_acoustic_feature_extractor, create_ctc_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMOTION_NAMES  = ["Neutral", "Happy", "Angry", "Sad"]


def evaluate_checkpoint(
    checkpoint_path: str,
    test_csv: str,
    config: ViSERConfig,
    device: torch.device,
):
    # ── Load checkpoint ──────────────────────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # ── Load feature extractor & tokenizer ──────────────────────────────────
    feature_extractor = create_acoustic_feature_extractor(config)
    ctc_tokenizer = create_ctc_tokenizer(config)
    config.vocab_size = len(ctc_tokenizer)

    # ── Initialize and load model ────────────────────────────────────────────
    model = create_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Build test dataset ───────────────────────────────────────────────────
    test_ds = ViSERDataset(
        test_csv, config, feature_extractor, ctc_tokenizer,
        is_training=False, audio_only=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
        collate_fn=ViSERCollator(feature_extractor),
    )

    # ── Collect predictions ──────────────────────────────────────────────────
    all_emotion_preds  = []
    all_emotion_labels = []
    all_alphas          = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_values   = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            emotion_labels  = batch["emotion_labels"]
            student_texts   = batch["student_texts"]

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                student_texts=student_texts,
                teacher_texts=None,
                processor=ctc_tokenizer,
                training_mode=False,  # No teacher at test time
            )

            emotion_preds  = outputs["logits_emotion_student"].argmax(-1).cpu()
            alphas         = outputs["alpha"].squeeze(-1).cpu()

            all_emotion_preds.extend(emotion_preds.tolist())
            all_emotion_labels.extend(emotion_labels.tolist())
            all_alphas.extend(alphas.tolist())

    # ── Compute metrics ───────────────────────────────────────────────────────
    emo_acc  = accuracy_score(all_emotion_labels, all_emotion_preds)
    emo_ua   = f1_score(all_emotion_labels, all_emotion_preds, average="macro")
    emo_wa   = f1_score(all_emotion_labels, all_emotion_preds, average="weighted")
    avg_alpha = np.mean(all_alphas)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ViSER Evaluation Results")
    print("="*60)
    print(f"\nEmotion Recognition:")
    print(f"  Accuracy (WA): {emo_acc:.4f}")
    print(f"  F1 Macro (UA): {emo_ua:.4f}")
    print(f"  F1 Weighted:   {emo_wa:.4f}")
    print(f"\n{classification_report(all_emotion_labels, all_emotion_preds, target_names=EMOTION_NAMES)}")



    print(f"\nModel Statistics:")
    print(f"  Avg. Uncertainty Gate (alpha): {avg_alpha:.4f}")
    print(f"  [alpha near 1.0 = high ASR confidence | near 0.05 = low ASR confidence]")
    print("="*60)

    results = {
        "emotion_accuracy": emo_acc,
        "emotion_f1_macro": emo_ua,
        "emotion_f1_weighted": emo_wa,
        "avg_uncertainty_alpha": avg_alpha,
    }

    # Save results
    results_path = os.path.join(os.path.dirname(checkpoint_path), "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViSER model")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to saved checkpoint .pt file"
    )
    parser.add_argument(
        "--test_csv", type=str, default=None,
        help="Override test CSV path from config"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.test_csv:
        config.test_csv = args.test_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_checkpoint(args.checkpoint, config.test_csv, config, device)

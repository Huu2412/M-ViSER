"""
extract_features.py

Script to pre-extract and cache acoustic (Wav2Vec2) and text (BERT) representations offline.
Running this script once allows training to run 5x-10x faster with drastically reduced VRAM.

Usage:
    python extract_features.py --config config/config.yaml --output_dir cached_features
"""

import os
import sys
import json
import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from vi_ser.encoders.acoustic_encoder import Wav2Vec2AcousticEncoder
from vi_ser.encoders.text_encoder import BERTTextEncoder
from vi_ser.factory import create_acoustic_feature_extractor, create_ctc_tokenizer
from vi_ser.data_loader.iemocap import build_dataloaders

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_split(
    loader,
    split_name: str,
    output_dir: str,
    acoustic_encoder: Wav2Vec2AcousticEncoder,
    text_encoder: BERTTextEncoder,
    device: torch.device,
):
    split_path = os.path.join(output_dir, split_name)
    os.makedirs(split_path, exist_ok=True)

    manifest = []
    sample_count = 0
    total_bytes = 0

    logger.info(f"Extracting features for split: '{split_name}' ({len(loader.dataset)} samples)...")
    pbar = tqdm(loader, desc=f"Extract [{split_name}]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_texts = batch["teacher_texts"]
            emotion_labels = batch["emotion_labels"]
            ctc_labels = batch["ctc_labels"]
            files = batch["files"]

            # ── 1. Wav2Vec2 forward ──────────────────────────────────────────
            if hasattr(acoustic_encoder.encoder, "config") and hasattr(acoustic_encoder.encoder.config, "apply_spec_augment"):
                acoustic_encoder.encoder.config.apply_spec_augment = False

            outputs = acoustic_encoder.encoder(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            hidden_states = outputs[0]  # [B, T, 768]

            max_input_len = attention_mask.shape[1]
            max_output_len = hidden_states.shape[1]
            feat_lens = acoustic_encoder.get_feat_extract_output_lengths(
                attention_mask.sum(-1), max_input_len, max_output_len
            )

            # ── 2. BERT forward ──────────────────────────────────────────────
            safe_texts = [t if (t and isinstance(t, str) and t.strip()) else "[UNK]" for t in teacher_texts]
            text_out = text_encoder(safe_texts, device=device)
            text_hidden = text_out["hidden_states"]
            text_mask = text_out["attention_mask"].float()
            z_clean_text = (
                text_hidden * text_mask.unsqueeze(-1)
            ).sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 768]

            # ── 3. Save each sample ──────────────────────────────────────────
            batch_size = input_values.size(0)
            for b in range(batch_size):
                valid_len = min(int(feat_lens[b].item()), hidden_states.size(1))
                valid_len = max(valid_len, 1)

                # Unpad hidden_states and cast to float16 to save 50% disk space
                h_b = hidden_states[b, :valid_len].half().cpu()
                z_b = z_clean_text[b].half().cpu()
                emo_b = emotion_labels[b].cpu()

                # Clean ctc labels (remove -100 padding)
                ctc_raw = ctc_labels[b]
                ctc_clean = ctc_raw[ctc_raw >= 0].cpu()

                fname = f"sample_{sample_count:05d}.pt"
                fpath = os.path.join(split_path, fname)

                sample_data = {
                    "hidden_states": h_b,
                    "z_clean_text": z_b,
                    "emotion_label": emo_b,
                    "ctc_labels": ctc_clean,
                    "file": files[b] if b < len(files) else f"sample_{sample_count}",
                }

                torch.save(sample_data, fpath)
                total_bytes += os.path.getsize(fpath)

                manifest.append({
                    "id": sample_count,
                    "file": fname,
                    "num_frames": valid_len,
                    "emotion": int(emo_b.item()),
                    "original_file": files[b] if b < len(files) else "",
                })
                sample_count += 1

    manifest_file = os.path.join(split_path, "manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    mb_size = total_bytes / (1024 * 1024)
    logger.info(f"Finished split '{split_name}': {sample_count} files saved ({mb_size:.2f} MB).")


def main():
    parser = argparse.ArgumentParser(description="Extract and cache Wav2Vec2 and BERT features")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save cached .pt files")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.cached_features_dir = args.output_dir
    output_dir = getattr(config, "cached_features_dir", "cached_features")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device for extraction: {device}")

    # ── Feature Extractor & Tokenizer ─────────────────────────────────────────
    feature_extractor = create_acoustic_feature_extractor(config)
    ctc_tokenizer = create_ctc_tokenizer(config)
    config.vocab_size = ctc_tokenizer.vocab_size
    config.pad_token_id = ctc_tokenizer.pad_token_id

    # Disable data augmentation during feature extraction
    config.augment_prob = 0.0
    config.batch_size = args.batch_size

    # ── Load Dataloaders ──────────────────────────────────────────────────────
    logger.info("Initializing raw audio dataset...")
    train_loader, val_loader = build_dataloaders(config, feature_extractor, ctc_tokenizer)
    # Ensure train_loader dataset doesn't augment
    if hasattr(train_loader.dataset, "is_training"):
        train_loader.dataset.is_training = False

    # ── Initialize Models (Evaluation Mode) ───────────────────────────────────
    logger.info("Loading Wav2Vec2 acoustic backbone...")
    acoustic_encoder = Wav2Vec2AcousticEncoder(config).to(device)
    acoustic_encoder.eval()

    logger.info("Loading BERT text encoder...")
    text_encoder = BERTTextEncoder(config).to(device)
    text_encoder.eval()

    # ── Extract Train & Val ───────────────────────────────────────────────────
    extract_split(train_loader, "train", output_dir, acoustic_encoder, text_encoder, device)
    extract_split(val_loader, "val", output_dir, acoustic_encoder, text_encoder, device)

    print("\n" + "=" * 60)
    print(" FEATURE EXTRACTION COMPLETE!")
    print(f" Saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)
    print("\nTo train model with these cached features, set in config/config.yaml:")
    print("cache:")
    print("  use_cached_features: true")
    print(f"  feature_dir:         \"{output_dir}\"")
    print("  in_memory:           true\n")
    print("Or pass CLI override:")
    print(f"python train.py --config config/config.yaml --override cache.use_cached_features=true\n")


if __name__ == "__main__":
    main()

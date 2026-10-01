"""
vi_ser/data_loader/cached_dataset.py

Ultra-fast DataLoader for pre-extracted Wav2Vec2 and BERT features.
Eliminates audio decoding, SpecAugment, and backbone forward passes during training.
"""

import os
import glob
import json
import logging
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CachedViSERDataset(Dataset):
    """
    Dataset that loads precomputed Wav2Vec2 hidden_states and BERT z_clean_text.
    Supports on-disk loading (.pt files) or ultra-fast in-memory caching.
    """

    def __init__(
        self,
        feature_dir: str,
        split: str = "train",
        in_memory: bool = False,
    ):
        super().__init__()
        self.split = split
        self.in_memory = in_memory
        self.split_dir = os.path.join(feature_dir, split)

        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(
                f"Feature directory for split '{split}' not found at: {self.split_dir}\n"
                f"Please run 'python extract_features.py --config config/config.yaml' first."
            )

        manifest_path = os.path.join(self.split_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_items = json.load(f)
            self.file_paths = [os.path.join(self.split_dir, item["file"]) for item in manifest_items]
        else:
            self.file_paths = sorted(glob.glob(os.path.join(self.split_dir, "*.pt")))

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .pt feature files found in {self.split_dir}")

        logger.info(f"Loaded {len(self.file_paths)} cached samples for split '{split}'.")

        self.memory_cache = None
        if self.in_memory:
            logger.info(f"Preloading {len(self.file_paths)} samples into RAM for 0-latency training...")
            self.memory_cache = []
            for p in self.file_paths:
                self.memory_cache.append(torch.load(p, map_location="cpu"))
            logger.info("Preload complete!")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict:
        if self.in_memory and self.memory_cache is not None:
            return self.memory_cache[idx]
        return torch.load(self.file_paths[idx], map_location="cpu")


class CachedViSERCollator:
    """
    Collator for cached features:
      - Pads hidden_states to batch max length [B, T_max, 768]
      - Constructs boolean audio_mask [B, T_max]
      - Stacks z_clean_text [B, 768]
      - Stacks emotion_labels [B]
      - Pads ctc_labels with -100 [B, L_max]
    """

    def __call__(self, batch: List[Dict]) -> Dict:
        B = len(batch)

        # ── 1. Pad Wav2Vec2 hidden_states [T_i, 768] ──────────────────────────
        hidden_states_list = [item["hidden_states"] for item in batch]
        max_T = max(h.size(0) for h in hidden_states_list)
        hidden_dim = hidden_states_list[0].size(-1)

        padded_hidden = torch.zeros(B, max_T, hidden_dim, dtype=torch.float32)
        audio_mask = torch.zeros(B, max_T, dtype=torch.bool)

        for i, h in enumerate(hidden_states_list):
            T_i = h.size(0)
            padded_hidden[i, :T_i] = h.float()
            audio_mask[i, :T_i] = True

        # ── 2. Stack BERT z_clean_text [768] ─────────────────────────────────
        z_clean_text_list = [item["z_clean_text"] for item in batch]
        z_clean_text = torch.stack([z.float() if isinstance(z, torch.Tensor) else torch.tensor(z, dtype=torch.float32) for z in z_clean_text_list])

        # ── 3. Stack Emotion Labels [B] ───────────────────────────────────────
        emotion_labels = torch.stack([
            item["emotion_label"] if isinstance(item["emotion_label"], torch.Tensor)
            else torch.tensor(item["emotion_label"], dtype=torch.long)
            for item in batch
        ])

        # ── 4. Pad CTC Labels [L_i] with -100 ────────────────────────────────
        ctc_list = [item.get("ctc_labels") for item in batch]
        padded_ctc = None
        if any(c is not None for c in ctc_list):
            max_L = max((len(c) for c in ctc_list if c is not None), default=0)
            padded = []
            for c in ctc_list:
                if c is None:
                    padded.append(torch.full((max_L,), -100, dtype=torch.long))
                else:
                    if not isinstance(c, torch.Tensor):
                        c = torch.tensor(c, dtype=torch.long)
                    pad_len = max_L - len(c)
                    padded.append(torch.cat([c, torch.full((pad_len,), -100, dtype=torch.long)]))
            padded_ctc = torch.stack(padded)

        # ── 5. Metadata ───────────────────────────────────────────────────────
        files = [item.get("file", f"sample_{i}") for i, item in enumerate(batch)]

        return {
            "hidden_states": padded_hidden,
            "audio_mask": audio_mask,
            "z_clean_text": z_clean_text,
            "emotion_labels": emotion_labels,
            "ctc_labels": padded_ctc,
            "files": files,
            "input_values": None,
            "attention_mask": None,
            "teacher_texts": None,
            "student_texts": None,
        }


def build_cached_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from cached features."""
    feature_dir = getattr(config, "cached_features_dir", "cached_features")
    in_memory = getattr(config, "cache_in_memory", False)

    train_ds = CachedViSERDataset(feature_dir=feature_dir, split="train", in_memory=in_memory)
    val_ds = CachedViSERDataset(feature_dir=feature_dir, split="val", in_memory=in_memory)

    collator = CachedViSERCollator()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader

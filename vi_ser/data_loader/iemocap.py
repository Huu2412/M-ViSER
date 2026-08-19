"""
vi_ser/data_loader/iemocap.py

Dataset class for M-ViSER training on IEMOCAP (or any English SER dataset).

Supports two modes:
  1. HuggingFace Dataset (hf_dataset in config) — auto 5-fold speaker-independent split
  2. CSV file (train_csv / val_csv in config) with columns:
       file      - path to audio file (.wav, 16kHz)
       text      - ground-truth transcript (used as teacher text)
       emotion   - emotion label (e.g. "neu", "hap", "ang", "sad")

Returns per sample:
  input_values, attention_mask, ctc_labels, emotion_label,
  teacher_text (ground-truth transcript), student_text (None → CTC decode online)
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ViSERDataset(Dataset):
    """
    Dataset for IEMOCAP / English SER training and evaluation.

    Loads audio, emotion label, and ground-truth transcript (teacher text).
    Teacher text is taken directly from the dataset (no external ASR needed).
    """

    def __init__(
        self,
        csv_path: str,
        config,
        feature_extractor,          # Wav2Vec2 AutoFeatureExtractor
        ctc_tokenizer,              # Wav2Vec2CTCTokenizer (English)
        teacher_cache=None,         # Unused — kept for API compatibility
        phowhisper_teacher=None,    # Unused — removed (was Vietnamese only)
        is_training: bool = True,
        audio_only: bool = False,   # Inference mode: no text labels
        hf_dataset=None,            # Hugging Face Dataset subset
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.ctc_tokenizer = ctc_tokenizer
        self.teacher_cache = teacher_cache
        self.phowhisper_teacher = phowhisper_teacher
        self.is_training = is_training
        self.audio_only = audio_only
        self.hf_dataset = hf_dataset

        if self.hf_dataset is not None:
            self.df = None
            logger.info(f"Loaded Hugging Face subset ({len(self.hf_dataset)} samples)")
        else:
            self.df = pd.read_csv(csv_path)
            logger.info(f"Loaded dataset: {csv_path} ({len(self.df)} samples)")
    
            # Validate required columns
            required = [config.speech_col, config.emotion_col]
            for col in required:
                assert col in self.df.columns, f"Missing column: {col}"

    def __len__(self):
        if self.hf_dataset is not None:
            return len(self.hf_dataset)
        return len(self.df)

    def _load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file, resample to 16kHz."""
        speech, sr = librosa.load(filepath, sr=self.config.sampling_rate)
        return speech, sr

    def _get_teacher_text(self, filepath: str, speech: np.ndarray) -> str:
        """Get teacher text: ground-truth from dataset (no ASR inference needed)."""
        # Fallback: return empty string; actual GT text is passed via clean_text in __getitem__
        return ""

    def _apply_augmentations(self, speech: np.ndarray, sr: int) -> np.ndarray:
        """Apply waveform augmentations (Pitch Shift, Noise Injection, Time Stretch)."""
        if not self.is_training or np.random.rand() > self.config.augment_prob:
            return speech
        
        weights = [
            self.config.augment_pitch_weight,
            self.config.augment_noise_weight,
            self.config.augment_time_weight
        ]
        sum_weights = sum(weights)
        if sum_weights <= 0:
            return speech
        weights = [w / sum_weights for w in weights]
        
        aug_choice = np.random.choice(["pitch", "noise", "time"], p=weights)
        
        if aug_choice == "pitch":
            n_steps = np.random.uniform(*self.config.augment_pitch_steps)
            speech = librosa.effects.pitch_shift(speech, sr=sr, n_steps=n_steps)
            
        elif aug_choice == "noise":
            noise_factor = np.random.uniform(*self.config.augment_noise_range)
            noise = np.random.randn(len(speech))
            speech = speech + noise_factor * noise
            
        elif aug_choice == "time":
            rate = np.random.uniform(*self.config.augment_time_range)
            speech = librosa.effects.time_stretch(speech, rate=rate)
            
        return speech

    # Map từ IEMOCAP major_emotion (English full names) sang label map keys
    _IEMOCAP_EMOTION_MAP = {
        "neutral":    "neu",
        "happy":      "hap",
        "excited":    "hap",   # excited → happy (gộp như chuẩn IEMOCAP 4-class)
        "angry":      "ang",
        "frustrated": "ang",   # frustrated → angry
        "sad":        "sad",
        "disgust":    "ang",
        "fear":       "sad",
        "surprise":   "hap",
    }

    def __getitem__(self, idx: int) -> Dict:
        if self.hf_dataset is not None:
            item_hf = self.hf_dataset[idx]

            # ── Audio: AbstractTTS/IEMOCAP dùng column 'audio' {bytes, path} ──
            audio_info = item_hf.get("audio") or item_hf.get("path")
            if not isinstance(audio_info, dict):
                audio_info = {
                    "path": getattr(audio_info, "path", f"audio_{idx}.wav"),
                    "bytes": getattr(audio_info, "bytes", None),
                    "array": getattr(audio_info, "array", None),
                    "sampling_rate": getattr(audio_info, "sampling_rate", 16000)
                }
            if audio_info.get("array") is not None:
                speech = np.array(audio_info["array"], dtype=np.float32)
                sr = audio_info.get("sampling_rate", 16000)
                # Normalize nếu là raw PCM integer (amplitude > 1)
                if np.abs(speech).max() > 1.0:
                    speech = speech / 32768.0
            elif audio_info.get("bytes") is not None:
                import io
                import soundfile as sf
                # dtype='float32' tự động normalize PCM int16 -> [-1, 1]
                # Đây là fix cho lỗi: sf.read() mặc định trả về int16 -> amplitude lên tới 32767!
                speech, sr = sf.read(io.BytesIO(audio_info["bytes"]), dtype='float32')
                speech = np.array(speech, dtype=np.float32)
                sr = int(sr)
            else:
                speech, sr = librosa.load(audio_info["path"], sr=self.config.sampling_rate)
            
            # Final safety clip: đảm bảo audio trong [-1, 1] trước khi vào Wav2Vec2
            max_amp = np.abs(speech).max()
            if max_amp > 1.0:
                speech = speech / max_amp

            if speech.ndim > 1:
                speech = speech.squeeze()

            # --- Chống lỗi NaN do audio quá ngắn / file hỏng ---
            if len(speech) < 400:  # < 25ms, quá ngắn để phân tích
                logger.warning(f"Audio quá ngắn hoặc rỗng tại {audio_info.get('path')}. Đang chèn audio tĩnh để chống lỗi NaN.")
                speech = np.zeros(self.config.sampling_rate, dtype=np.float32)

            filepath = audio_info.get("path") or item_hf.get("file", f"hf_audio_{idx}")

            # ── Emotion: hỗ trợ cả 'emotion', 'major_emotion' (IEMOCAP) ──────
            raw_emotion = (
                item_hf.get("emotion")
                or item_hf.get("major_emotion")
                or "neutral"
            )
            raw_emotion = str(raw_emotion).lower().strip()
            # Nếu đã là short-code (neu/hap/ang/sad) thì giữ nguyên
            if raw_emotion not in self.config.emotion_label_map:
                raw_emotion = self._IEMOCAP_EMOTION_MAP.get(raw_emotion, "neu")
            emotion_str = raw_emotion

            # ── Text: hỗ trợ 'text', 'transcription' (IEMOCAP) ──────────────
            clean_text = str(
                item_hf.get("text")
                or item_hf.get("transcription")
                or ""
            )
        else:
            row = self.df.iloc[idx]
            filepath = row[self.config.speech_col]
            speech, sr = self._load_audio(filepath)
            
            emotion_str = str(row[self.config.emotion_col])
            clean_text = str(row.get(self.config.text_col, ""))
            
        if sr != self.config.sampling_rate:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=self.config.sampling_rate)
            sr = self.config.sampling_rate
        
        # ── Augmentation ───────────────────────────────────────────────────
        speech = self._apply_augmentations(speech, sr)

        # Filter by max duration
        max_len = int(self.config.max_audio_length_sec * self.config.sampling_rate)
        speech = speech[:max_len]

        # ── Normalize audio ─────────────────────────────────────────────────
        # Wav2Vec2 kỳ vọng waveform ~[-1, 1]. Nếu amplitude quá lớn hoặc có
        # NaN/Inf (từ augmentation hoặc file hỏng) → Wav2Vec2 sẽ sinh NaN hidden states.
        speech = np.nan_to_num(speech, nan=0.0, posinf=1.0, neginf=-1.0)
        max_abs = np.abs(speech).max()
        if max_abs > 1e-6:  # tránh chia cho 0 với audio im lặng
            speech = speech / max_abs  # peak normalize về [-1, 1]
        else:
            # Audio im lặng hoàn toàn: thêm tiny noise để tránh all-zero tensor
            speech = np.random.randn(len(speech)).astype(np.float32) * 1e-6

        # ── Feature extraction (ViP-VL processor) ──────────────────────────
        inputs = self.feature_extractor(
            speech,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)  # [T]

        item = {
            "input_values": input_values,
            "file": filepath,
        }

        if not self.audio_only:
            # ── Emotion label ───────────────────────────────────────────────
            item["emotion_label"] = torch.tensor(
                self.config.emotion_label_map[emotion_str], dtype=torch.long
            )


            # ── Teacher text (PhoWhisper transcription) ────────────────────
            if self.is_training:
                teacher_text = self._get_teacher_text(filepath, speech)
                item["teacher_text"] = teacher_text if teacher_text else clean_text
            else:
                item["teacher_text"] = clean_text

            # ── CTC labels (clean text → fallback to teacher_text) ─────────
            target_text_for_ctc = clean_text if clean_text else item["teacher_text"]
            
            if target_text_for_ctc and self.ctc_tokenizer is not None:
                ctc_ids = self.ctc_tokenizer(target_text_for_ctc).input_ids
                if len(ctc_ids) == 0:
                    ctc_ids = [self.ctc_tokenizer.pad_token_id]
                item["ctc_labels"] = torch.tensor(ctc_ids, dtype=torch.long)
            else:
                item["ctc_labels"] = torch.tensor([self.ctc_tokenizer.pad_token_id], dtype=torch.long)
            
            # Luôn set student_text = None để model.py tự động decode_ctc (Bắt buộc cho Repair Gate)
            item["student_text"] = None

        return item


class ViSERCollator:
    """
    Dynamic padding collator for ViSER batches.
    Pads audio to batch max length, pads CTC labels to batch max label length.
    """

    def __init__(self, feature_extractor, pad_token_id: int = 0):
        self.feature_extractor = feature_extractor
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict:
        # ── Pad audio ─────────────────────────────────────────────────────
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # ── Collect text fields ───────────────────────────────────────────
        batch["files"]         = [f.get("file", "") for f in features]
        batch["student_texts"] = [f.get("student_text", "") for f in features]
        batch["teacher_texts"] = [f.get("teacher_text", "") for f in features]

        # ── Pad labels ───────────────────────────────────────────────────
        if "emotion_label" in features[0]:
            batch["emotion_labels"]  = torch.stack([f["emotion_label"]  for f in features])

        # ── Pad CTC labels (fill with -100 for ignored positions) ─────────
        ctc_labels_list = [f.get("ctc_labels") for f in features]
        if any(c is not None for c in ctc_labels_list):
            max_label_len = max(
                (len(c) for c in ctc_labels_list if c is not None), default=0
            )
            padded = []
            for c in ctc_labels_list:
                if c is None:
                    padded.append(torch.full((max_label_len,), -100, dtype=torch.long))
                else:
                    pad_len = max_label_len - len(c)
                    padded.append(
                        torch.cat([c, torch.full((pad_len,), -100, dtype=torch.long)])
                    )
            batch["ctc_labels"] = torch.stack(padded)

        return batch


def build_dataloaders(config, feature_extractor, ctc_tokenizer, teacher_cache=None, phowhisper_teacher=None):
    """Build train/val/test DataLoaders."""
    
    if getattr(config, "hf_dataset", None):
        import datasets
        from sklearn.model_selection import GroupKFold
        import pandas as pd
        import re

        logger.info(f"Loading HF dataset {config.hf_dataset} ...")
        ds = datasets.load_dataset(config.hf_dataset, split="train")

        # Cast audio column về decode=False (raw bytes) — hỗ trợ cả 'audio' và 'path'
        audio_col = "audio" if "audio" in ds.column_names else "path"
        ds = ds.cast_column(audio_col, datasets.Audio(decode=False))

        # ── Xác định speaker_id để split speaker-independent ────────────────
        if "speaker_id" in ds.column_names:
            speaker_ids = ds["speaker_id"]
        elif "speaker" in ds.column_names:
            speaker_ids = ds["speaker"]
        else:
            # AbstractTTS/IEMOCAP: extract speaker từ filename  Ses01F_impro01_F000 → '1F'
            file_col = "file" if "file" in ds.column_names else None
            if file_col:
                def extract_speaker(fname):
                    m = re.match(r"Ses(\d+[MF])", str(fname))
                    return m.group(1) if m else str(fname)[:4]
                speaker_ids = [extract_speaker(f) for f in ds[file_col]]
            else:
                speaker_ids = [str(i % 10) for i in range(len(ds))]
                logger.warning("No speaker column found — using dummy speaker IDs.")

        df = pd.DataFrame({"speaker_id": speaker_ids})
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(df, groups=df["speaker_id"]))

        fold_idx = config.current_fold - 1
        train_idx, val_idx = splits[fold_idx]
        logger.info(f"Fold {config.current_fold}: train={len(train_idx)}, val={len(val_idx)}")

        train_hf_ds = ds.select(train_idx)
        val_hf_ds = ds.select(val_idx)
        
        train_ds = ViSERDataset(
            csv_path=None, config=config, feature_extractor=feature_extractor, ctc_tokenizer=ctc_tokenizer,
            teacher_cache=teacher_cache, phowhisper_teacher=phowhisper_teacher, is_training=True,
            hf_dataset=train_hf_ds
        )
        val_ds = ViSERDataset(
            csv_path=None, config=config, feature_extractor=feature_extractor, ctc_tokenizer=ctc_tokenizer,
            teacher_cache=teacher_cache, phowhisper_teacher=phowhisper_teacher, is_training=False,
            hf_dataset=val_hf_ds
        )
    else:
        train_ds = ViSERDataset(
            config.train_csv, config, feature_extractor, ctc_tokenizer,
            teacher_cache=teacher_cache, phowhisper_teacher=phowhisper_teacher,
            is_training=True,
        )
        val_ds = ViSERDataset(
            config.val_csv, config, feature_extractor, ctc_tokenizer,
            teacher_cache=teacher_cache, phowhisper_teacher=phowhisper_teacher,
            is_training=False,
        )

    collator = ViSERCollator(feature_extractor)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collator,
    )
    return train_loader, val_loader

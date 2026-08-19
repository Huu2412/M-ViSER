"""
vi_ser/encoders/acoustic_encoder.py

Wav2Vec2 Acoustic Encoder with CTC head (Student ASR auxiliary task from MTL-SER).
Provides:
  - hidden_states: [B, T, H]  (frame-level for CTC loss)
  - z_audio:       [B, D]     (mean-pooled utterance embedding → fusion_dim)
  - logits_ctc:    [B, T, V]  (CTC logits for student ASR)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor

logger = logging.getLogger(__name__)


def _safe_normalize(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Replace NaN/Inf with zeros. Log a warning if any are found."""
    if not torch.isfinite(tensor).all():
        logger.warning(
            f"NaN/Inf detected in {name} "
            f"(nan={tensor.isnan().sum().item()}, "
            f"inf={tensor.isinf().sum().item()}). Replacing with zeros."
        )
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


class SafeLayerNorm(nn.LayerNorm):
    """
    Fallback to standard nn.LayerNorm as PyTorch's eps already handles zero-variance.
    Random noise injection was removed to ensure reproducibility.
    """
    pass


class Wav2Vec2AcousticEncoder(nn.Module):
    """
    Wraps the configured Wav2Vec2 acoustic model (e.g. facebook/wav2vec2-base-960h)
    as the acoustic backbone. Also contains a CTC head that mirrors the student ASR
    auxiliary task in MTL-SER.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.acoustic_hidden_size

        # ── Load Wav2Vec2 backbone ───────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(
            config.acoustic_model_name,
            cache_dir=config.cache_dir,
            apply_spec_augment=False,  # CRITICAL: Prevent zero-variance NaN on short audio
            use_safetensors=False,
            attn_implementation="eager",  # CRITICAL: Fixes Wav2Vec2 SDPA NaN bug on padded tokens
        )

        # Freeze CNN feature extractor (same as MTL-SER approach)
        if config.freeze_feature_extractor:
            self._freeze_feature_extractor()

        # Freeze all transformer layers (optional)
        if config.freeze_acoustic_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ── CTC Head (Student ASR) ───────────────────────────────────────────
        self.dropout = nn.Dropout(config.dropout)
        self.ctc_head = nn.Linear(self.hidden_size, config.vocab_size)

        # ── Projection to fusion_dim for downstream AURORA fusion ────────────
        # Dùng SafeLayerNorm thay LayerNorm để chống NaN khi input all-zero
        self.audio_proj = nn.Sequential(
            nn.Linear(self.hidden_size, config.fusion_dim),
            nn.GELU(),
            SafeLayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
        )

    def _freeze_feature_extractor(self):
        """Freeze CNN feature extractor layers (same as Wav2Vec2)."""
        if hasattr(self.encoder, "freeze_feature_encoder"):
            self.encoder.freeze_feature_encoder()
        elif hasattr(self.encoder, "feature_extractor"):
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False
        elif hasattr(self.encoder, "encoder") and hasattr(
            self.encoder.encoder, "feature_extractor"
        ):
            for param in self.encoder.encoder.feature_extractor.parameters():
                param.requires_grad = False

    def get_feat_extract_output_lengths(
        self,
        input_lengths: torch.LongTensor,
        max_input_len: int = None,
        max_output_len: int = None,
    ):
        """
        Compute CTC output lengths from input audio lengths.
        Always prioritizes HF's accurate `_get_feat_extract_output_lengths`.
        """
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            # PyTorch `transformers` library usually expects a Tensor now because it uses `torch.div` internally.
            # Passing a numpy array will crash `torch.div`.
            if isinstance(input_lengths, torch.Tensor):
                out_lens = self.encoder._get_feat_extract_output_lengths(input_lengths)
                # Đảm bảo kết quả trả về là tensor
                if not isinstance(out_lens, torch.Tensor):
                    lengths = torch.tensor(out_lens, device=input_lengths.device, dtype=torch.long)
                else:
                    lengths = out_lens.to(input_lengths.device).long()
            else:
                lengths = self.encoder._get_feat_extract_output_lengths(input_lengths)
                
            if max_output_len is not None:
                return lengths.clamp(min=1, max=max_output_len)
            return lengths

        # Fallback ratio approximation
        if max_input_len is not None and max_output_len is not None:
            lengths = torch.round(
                input_lengths.float() * (max_output_len / max_input_len)
            ).long()
            return lengths.clamp(min=1, max=max_output_len)

        return (input_lengths - 1) // 320 + 1

    def _sanitize_audio(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Removes NaN/Inf from audio input before Wav2Vec2.
        NOTE: Do NOT re-normalize here. input_values come from Wav2Vec2FeatureExtractor
        which applies (x-mean)/std — values in [-5, +20] are NORMAL and expected.
        Re-normalizing to [-1,1] would corrupt the Wav2Vec2 input and cause NaN!
        """
        # Only fix NaN/Inf — do NOT touch magnitude
        if not torch.isfinite(input_values).all():
            n_bad = (~torch.isfinite(input_values)).sum().item()
            logger.warning(f"{n_bad} NaN/Inf value(s) in input_values. Replacing with 0.")
            input_values = torch.nan_to_num(input_values, nan=0.0, posinf=0.0, neginf=0.0)
        return input_values

    def forward(
        self,
        input_values: torch.Tensor,         # [B, T_audio]
        attention_mask: torch.Tensor = None,
        output_hidden_states: bool = False,
    ):
        """
        Returns:
            hidden_states: [B, T, H]  - frame-level encoder output
            z_audio:       [B, fusion_dim] - mean-pooled + projected
            logits_ctc:    [B, T, vocab_size] - CTC logits
        """
        # ── Guard 1: sanitize & normalize input audio ────────────────────────
        input_values = self._sanitize_audio(input_values)

        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # Last layer hidden states: [B, T, H]
        hidden_states = outputs[0].float()  # cast to float32

        # ── Guard 2: sanitize hidden_states từ Wav2Vec2 ──────────────────────
        hidden_states = _safe_normalize(hidden_states, "Wav2Vec2 hidden_states")

        hidden_states = self.dropout(hidden_states)

        # CTC logits (Student ASR)
        logits_ctc = self.ctc_head(hidden_states)  # [B, T, V]
        logits_ctc = _safe_normalize(logits_ctc, "logits_ctc")

        # ── Mean pool over time → utterance embedding ────────────────────────
        if attention_mask is not None:
            max_input_len = attention_mask.shape[1]
            max_output_len = hidden_states.shape[1]
            feat_len = self.get_feat_extract_output_lengths(
                attention_mask.sum(-1), max_input_len, max_output_len
            )
            mask = torch.zeros(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
            for i, length in enumerate(feat_len):
                length = min(length.item(), max_output_len)
                length = max(length, 1)  # đảm bảo ít nhất 1 frame
                mask[i, :length] = True

            # Masked mean — clamp(min=1) tránh chia cho 0
            hidden_masked = hidden_states * mask.unsqueeze(-1)
            z_audio = hidden_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
            z_audio = hidden_states.mean(dim=1)  # [B, H]

        # ── Guard 3: sanitize z_audio TRƯỚC audio_proj ──────────────────────
        # QUAN TRỌNG: SafeLayerNorm xử lý zero-variance bên trong audio_proj,
        # nhưng vẫn cần clean NaN/Inf có thể xuất hiện sau mean-pool.
        z_audio = _safe_normalize(z_audio, "z_audio (pre-proj)")

        # ── Project to fusion_dim ────────────────────────────────────────────
        z_audio = self.audio_proj(z_audio)  # [B, fusion_dim]

        # ── Guard 4: final check sau audio_proj ─────────────────────────────
        z_audio = _safe_normalize(z_audio, "z_audio (post-proj)")

        return {
            "hidden_states": hidden_states,    # [B, T, H]
            "audio_mask": mask,                # [B, T] (True=valid, False=padding)
            "z_audio": z_audio,                # [B, fusion_dim]
            "logits_ctc": logits_ctc,          # [B, T, vocab_size]
        }


class AcousticFeatureExtractor:
    """Helper to load Wav2Vec2 feature extractor / processor."""

    @staticmethod
    def from_pretrained(model_name: str, cache_dir: str = None):
        try:
            return AutoFeatureExtractor.from_pretrained(
                model_name, cache_dir=cache_dir
            )
        except OSError:
            logger.warning(
                f"Feature extractor config not found for '{model_name}'. "
                "Falling back to 'facebook/wav2vec2-base'."
            )
            from transformers import Wav2Vec2FeatureExtractor
            return Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base", cache_dir=cache_dir
            )

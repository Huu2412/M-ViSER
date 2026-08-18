"""
vi_ser/encoders/acoustic_encoder.py

Wav2Vec2 Acoustic Encoder with CTC head (Student ASR auxiliary task from MTL-SER).
Provides:
  - hidden_states: [B, T, H]  (frame-level for CTC loss)
  - z_audio:       [B, D]     (mean-pooled utterance embedding → fusion_dim)
  - logits_ctc:    [B, T, V]  (CTC logits for student ASR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor


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

        # ── Load ViP-VL backbone ─────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(
            config.acoustic_model_name,
            cache_dir=config.cache_dir,
            use_safetensors=False,
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
        self.audio_proj = nn.Sequential(
            nn.Linear(self.hidden_size, config.fusion_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
        )

    def _freeze_feature_extractor(self):
        """Freeze CNN feature extractor layers (same as Wav2Vec2)."""
        # ViP-VL exposes feature_extractor or encoder.feature_extractor
        if hasattr(self.encoder, "feature_extractor"):
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False
        elif hasattr(self.encoder, "encoder") and hasattr(
            self.encoder.encoder, "feature_extractor"
        ):
            for param in self.encoder.encoder.feature_extractor.parameters():
                param.requires_grad = False

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor, max_input_len: int = None, max_output_len: int = None):
        """
        Compute CTC output lengths from input audio lengths.
        To ensure 100% accuracy regardless of backbone (Wav2Vec2 vs ChunkFormer),
        we use the dynamic ratio: (actual_audio_len / max_audio_len) * max_feat_len
        """
        if max_input_len is not None and max_output_len is not None:
            return torch.round(input_lengths.float() * (max_output_len / max_input_len)).long()
            
        # Fallback if dimensions are not provided
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(input_lengths)
        return (input_lengths - 1) // 320 + 1

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
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # Last layer hidden states: [B, T, H]
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # CTC logits (Student ASR) - same as MTL-SER lm_head
        logits_ctc = self.ctc_head(hidden_states)  # [B, T, V]

        # Mean pool over time → utterance embedding → project to fusion_dim
        if attention_mask is not None:
            # Mask padding before pooling (use dynamic ratio for 100% accuracy)
            max_input_len = attention_mask.shape[1]
            max_output_len = hidden_states.shape[1]
            feat_len = self.get_feat_extract_output_lengths(
                attention_mask.sum(-1), max_input_len, max_output_len
            )
            mask = torch.zeros(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
            for i, length in enumerate(feat_len):
                # Ensure length does not exceed max_output_len
                length = min(length.item(), max_output_len)
                mask[i, :length] = True
            # Masked mean
            hidden_masked = hidden_states * mask.unsqueeze(-1)
            z_audio = hidden_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            z_audio = hidden_states.mean(dim=1)  # [B, H]

        z_audio = self.audio_proj(z_audio)  # [B, fusion_dim]

        return {
            "hidden_states": hidden_states,    # [B, T, H]
            "z_audio": z_audio,                 # [B, fusion_dim]
            "logits_ctc": logits_ctc,           # [B, T, vocab_size]
        }


class AcousticFeatureExtractor:
    """Helper to load ViP-VL feature extractor / processor."""

    @staticmethod
    def from_pretrained(model_name: str, cache_dir: str = None):
        try:
            return AutoFeatureExtractor.from_pretrained(
                model_name, cache_dir=cache_dir
            )
        except OSError:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Feature extractor config not found for '{model_name}'. Falling back to 'facebook/wav2vec2-base'.")
            from transformers import Wav2Vec2FeatureExtractor
            return Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base", cache_dir=cache_dir
            )

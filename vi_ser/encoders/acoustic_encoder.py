"""
vi_ser/acoustic_encoder.py

ViP-VL Acoustic Encoder with CTC head (Student ASR).
ViP-VL uses ChunkFormer architecture loaded via AutoModel.
Provides:
  - hidden_states: [B, T, H]  (frame-level for CTC)
  - z_audio:       [B, D]     (mean-pooled utterance embedding)
  - logits_ctc:    [B, T, V]  (CTC logits for student ASR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor


class VipVlAcousticEncoder(nn.Module):
    """
    Wraps ViP-VL (khanhld/vip-vl-base-vie) as the acoustic backbone.
    Also contains a CTC head that mirrors the student ASR task in MTL-SER.
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
            nn.ReLU(),
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

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Compute CTC output lengths from input audio lengths.
        Falls back to Wav2Vec2-style stride computation.
        """
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(input_lengths)
        # Approximate for ViP-VL (ChunkFormer stride ~ 320 at 16kHz)
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
            # Mask padding before pooling
            feat_len = self.get_feat_extract_output_lengths(attention_mask.sum(-1))
            mask = torch.zeros(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
            for i, length in enumerate(feat_len):
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

"""
vi_ser/cross_modal.py

Cross-Modal Encoders.
Projects audio and text embeddings into a shared fusion_dim space.
Inspired by AURORA CrossModalEncoders.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SafeLayerNorm(nn.LayerNorm):
    """LayerNorm với guard chống zero-variance để tránh NaN."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=-1, keepdim=True)
        if (std < 1e-6).any():
            x = x + torch.randn_like(x) * 1e-6
        return super().forward(x)


class CrossModalEncoders(nn.Module):
    """
    Dual linear encoders that project audio and text embeddings
    into the same fusion_dim space for cross-modal interaction.

    Input:
        z_audio: [B, audio_dim]  (from Wav2Vec2 audio_proj)
        z_text:  [B, text_dim]   (from BERT [CLS] proj)

    Output:
        z_audio_enc: [B, fusion_dim]
        z_text_enc:  [B, fusion_dim]
    """

    def __init__(
        self,
        audio_input_dim: int,
        text_input_dim: int,
        fusion_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_input_dim, fusion_dim),
            nn.GELU(),
            SafeLayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.GELU(),
            SafeLayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_audio: torch.Tensor,  # [B, audio_input_dim]
        z_text: torch.Tensor,   # [B, text_input_dim]
    ):
        z_audio_enc = self.audio_encoder(z_audio)   # [B, fusion_dim]
        z_text_enc = self.text_encoder(z_text)      # [B, fusion_dim]

        # Guard sau cross-modal projection
        if not torch.isfinite(z_audio_enc).all():
            logger.warning("NaN/Inf in z_audio_enc after CrossModalEncoder. Replacing with zeros.")
            z_audio_enc = torch.nan_to_num(z_audio_enc, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(z_text_enc).all():
            logger.warning("NaN/Inf in z_text_enc after CrossModalEncoder. Replacing with zeros.")
            z_text_enc = torch.nan_to_num(z_text_enc, nan=0.0, posinf=0.0, neginf=0.0)

        return z_audio_enc, z_text_enc

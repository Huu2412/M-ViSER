"""
vi_ser/cross_modal.py

Cross-Modal Encoders.
Projects audio and text embeddings into a shared fusion_dim space.
Inspired by AURORA CrossModalEncoders.
"""

import torch
import torch.nn as nn


class CrossModalEncoders(nn.Module):
    """
    Dual linear encoders that project audio and text embeddings
    into the same fusion_dim space for cross-modal interaction.

    Input:
        z_audio: [B, audio_dim]  (from ViP-VL audio_proj OR raw hidden_size)
        z_text:  [B, text_dim]   (from PhoBERT proj)

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
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_audio: torch.Tensor,  # [B, audio_input_dim]
        z_text: torch.Tensor,   # [B, text_input_dim]
    ):
        z_audio_enc = self.audio_encoder(z_audio)   # [B, fusion_dim]
        z_text_enc = self.text_encoder(z_text)      # [B, fusion_dim]
        return z_audio_enc, z_text_enc

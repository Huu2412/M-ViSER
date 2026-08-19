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
    Bidirectional Multi-Head Cross-Attention for Audio-Text Fusion (AURORA).

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
        num_heads: int = 8,
    ):
        super().__init__()

        # 1. Linear projections to shared space
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_input_dim, fusion_dim),
            nn.GELU(),
            SafeLayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.GELU(),
            SafeLayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        # 2. Bidirectional Cross-Attention
        # Audio attends to Text (Query=Audio, Key/Value=Text)
        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.audio_norm = SafeLayerNorm(fusion_dim)
        
        # Text attends to Audio (Query=Text, Key/Value=Audio)
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.text_norm = SafeLayerNorm(fusion_dim)

    def forward(
        self,
        z_audio: torch.Tensor,  # [B, audio_input_dim]
        z_text: torch.Tensor,   # [B, text_input_dim]
    ):
        # Project inputs to fusion_dim
        z_a = self.audio_proj(z_audio)   # [B, fusion_dim]
        z_t = self.text_proj(z_text)     # [B, fusion_dim]

        # Reshape to sequence length of 1 for MultiheadAttention
        # [B, 1, fusion_dim]
        z_a_seq = z_a.unsqueeze(1)
        z_t_seq = z_t.unsqueeze(1)

        # Cross-Attention: Audio attends to Text
        attn_a, _ = self.audio_cross_attn(query=z_a_seq, key=z_t_seq, value=z_t_seq)
        
        # Cross-Attention: Text attends to Audio
        attn_t, _ = self.text_cross_attn(query=z_t_seq, key=z_a_seq, value=z_a_seq)

        # Residual connection and LayerNorm
        z_audio_enc = self.audio_norm(z_a_seq + attn_a).squeeze(1)  # [B, fusion_dim]
        z_text_enc = self.text_norm(z_t_seq + attn_t).squeeze(1)    # [B, fusion_dim]

        # Guard sau cross-modal projection
        if not torch.isfinite(z_audio_enc).all():
            logger.warning("NaN/Inf in z_audio_enc after CrossModalEncoder. Replacing with zeros.")
            z_audio_enc = torch.nan_to_num(z_audio_enc, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(z_text_enc).all():
            logger.warning("NaN/Inf in z_text_enc after CrossModalEncoder. Replacing with zeros.")
            z_text_enc = torch.nan_to_num(z_text_enc, nan=0.0, posinf=0.0, neginf=0.0)

        return z_audio_enc, z_text_enc

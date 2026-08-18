"""
vi_ser/audio_guided_gmu.py

Audio-Guided Gated Multimodal Unit (GMU).
Uses acoustic features to gate the fusion of audio and (repaired) text embeddings.
Inspired by AURORA AudioGuidedGatedFusion.

The key insight: audio features determine HOW MUCH to trust the text branch.
"""

import torch
import torch.nn as nn


class AudioGuidedGatedFusion(nn.Module):
    """
    Audio-Guided Gated Multimodal Unit.

    The gate is computed from the audio embedding (audio-centric fusion):
        g = sigmoid(audio_proj(z_audio))         -- audio gate
        z_fused = g ⊙ audio_proj(z_audio) + (1-g) ⊙ text_proj(z_text_repaired)

    Additionally, the UncertaintyGate alpha scales the text contribution:
        text_contribution = (1 - g) * alpha * text_proj(z_text_repaired)

    Then applies a 2-layer FFN with residual + LayerNorm for refinement.
    """

    def __init__(
        self,
        fusion_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Audio-driven gate (computed from audio features)
        self.audio_gate_proj = nn.Linear(fusion_dim, fusion_dim)

        # Projection layers (both inputs already at fusion_dim)
        self.audio_proj = nn.Linear(fusion_dim, fusion_dim)
        self.text_proj  = nn.Linear(fusion_dim, fusion_dim)

        # Feed-forward refinement network
        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        z_audio: torch.Tensor,         # [B, fusion_dim]
        z_text_repaired: torch.Tensor,  # [B, fusion_dim] — repaired ASR embedding
        alpha: torch.Tensor,            # [B, 1] — uncertainty gate from UncertaintyGate
    ) -> torch.Tensor:
        """
        Returns:
            z_fused: [B, fusion_dim]
        """
        # Audio-driven gate: how much to use audio vs text
        g = torch.sigmoid(self.audio_gate_proj(z_audio))  # [B, fusion_dim]

        # Project both modalities
        audio_feat = self.audio_proj(z_audio)              # [B, fusion_dim]
        text_feat  = self.text_proj(z_text_repaired)       # [B, fusion_dim]

        # Gated fusion: alpha scales the text contribution
        # g controls audio prominence; alpha controls ASR confidence
        fused = g * audio_feat + (1 - g) * alpha * text_feat  # [B, fusion_dim]

        # FFN refinement with residual connection
        refined = self.fusion_ffn(fused)                   # [B, fusion_dim]
        z_fused = self.norm(fused + refined)               # residual + norm

        return z_fused

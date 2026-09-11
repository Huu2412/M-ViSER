"""
vi_ser/audio_guided_gmu.py

Audio-Guided Gated Multimodal Unit (GMU).
Uses acoustic features to gate the fusion of audio and (repaired) text embeddings.
Inspired by AURORA AudioGuidedGatedFusion.

The key insight: audio features determine HOW MUCH to trust the text branch.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SafeLayerNorm(nn.LayerNorm):
    """
    Fallback to standard nn.LayerNorm as PyTorch's eps already handles zero-variance.
    Random noise injection was removed to ensure reproducibility.
    """
    pass


class AudioGuidedGatedFusion(nn.Module):
    """
    Audio-Guided Gated Multimodal Unit.

    The gate is computed from both modalities to decide trust levels:
        g = sigmoid(audio_gate_proj(z_audio) + text_gate_proj(z_text_repaired))
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

        # Gate projections (both modalities)
        self.audio_gate_proj = nn.Linear(fusion_dim, fusion_dim)
        self.text_gate_proj = nn.Linear(fusion_dim, fusion_dim)

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

        # SafeLayerNorm để tránh NaN khi fused vector gần all-zero
        self.norm = SafeLayerNorm(fusion_dim)

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
        # Gate: how much to use audio vs text (informed by both modalities)
        g = torch.sigmoid(self.audio_gate_proj(z_audio) + self.text_gate_proj(z_text_repaired))  # [B, fusion_dim]

        # Project both modalities
        audio_feat = self.audio_proj(z_audio)              # [B, fusion_dim]
        text_feat  = self.text_proj(z_text_repaired)       # [B, fusion_dim]

        # Gated fusion: alpha scales the text contribution
        # g controls audio prominence; alpha controls ASR confidence
        fused = g * audio_feat + (1 - g) * alpha * text_feat  # [B, fusion_dim]

        # Guard trước FFN: tránh NaN propagation vào refinement
        if not torch.isfinite(fused).all():
            logger.warning("NaN/Inf in fused before FFN. Replacing with zeros.")
            fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        # FFN refinement with residual connection
        refined = self.fusion_ffn(fused)                   # [B, fusion_dim]

        # Guard trước LayerNorm
        if not torch.isfinite(refined).all():
            logger.warning("NaN/Inf in refined after FFN. Replacing with zeros.")
            refined = torch.nan_to_num(refined, nan=0.0, posinf=0.0, neginf=0.0)

        z_fused = self.norm(fused + refined)               # residual + norm

        # Final guard
        if not torch.isfinite(z_fused).all():
            logger.warning("NaN/Inf in z_fused after GMU norm. Replacing with zeros.")
            z_fused = torch.nan_to_num(z_fused, nan=0.0, posinf=0.0, neginf=0.0)

        return z_fused


class AuroraGMU(nn.Module):
    """
    Faithful port of AURORA's AudioGuidedGatedFusion.

    Key difference from AudioGuidedGatedFusion:
      - Signature: forward(text_feat, audio_feat) — NO alpha parameter.
      - Teacher path always has full confidence (clean GT text), so alpha is not needed.
      - Fusion logic mirrors AURORA exactly:
          f_t  = text_proj(text_feat)
          f_a  = audio_proj(audio_feat)
          f_at = f_a + f_t                          (additive combination)
          f_at_prime = fusion_ffn(f_at) + f_at       (residual FFN)
          w    = sigmoid(gate_audio(f_a) + gate_fusion(f_at_prime))
          fused = (1 - w) * f_a + w * f_at_prime
          out   = out_proj(fused)

    Used exclusively by the Teacher path.
    """

    def __init__(
        self,
        fusion_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.audio_proj = nn.Linear(fusion_dim, fusion_dim)
        self.text_proj  = nn.Linear(fusion_dim, fusion_dim)

        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.gate_audio   = nn.Linear(fusion_dim, fusion_dim)
        self.gate_fusion  = nn.Linear(fusion_dim, fusion_dim)

        self.out_proj = nn.Linear(fusion_dim, fusion_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        text_feat: torch.Tensor,   # [B, fusion_dim] — z_clean_text from cross-modal
        audio_feat: torch.Tensor,  # [B, fusion_dim] — z_audio from cross-modal
    ) -> torch.Tensor:
        """
        Returns:
            fused: [B, fusion_dim]
        """
        f_t = self.text_proj(text_feat)    # [B, fusion_dim]
        f_a = self.audio_proj(audio_feat)  # [B, fusion_dim]

        # Additive combination (AURORA style)
        f_at = f_a + f_t                                  # [B, fusion_dim]

        # Residual FFN refinement
        f_at_prime = self.fusion_ffn(f_at) + f_at         # [B, fusion_dim]

        # Audio-guided gate
        gate_pre = self.gate_audio(f_a) + self.gate_fusion(f_at_prime)
        w = self.sigmoid(gate_pre)                        # [B, fusion_dim]

        # Gated fusion: audio baseline + text-enriched contribution
        fused = (1.0 - w) * f_a + w * f_at_prime         # [B, fusion_dim]
        fused = self.out_proj(fused)                       # [B, fusion_dim]

        return fused

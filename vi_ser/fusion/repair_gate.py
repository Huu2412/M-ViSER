"""
vi_ser/repair_gate.py

Repair Gate modules from AURORA:
  - RepairMLP: Computes bounded delta to correct noisy ASR text embedding
  - UncertaintyGate: Computes α to weight student ASR reliability

Both modules use [z_audio; z_asr] concatenation as input.
"""

import torch
import torch.nn as nn
from typing import Optional


class RepairMLP(nn.Module):
    """
    ASR Error Repair via residual correction in text embedding space.

    Formula:
        delta_raw = MLP([z_audio; z_asr_text])   -> [B, fusion_dim]
        delta     = delta_scale * tanh(delta_raw)  (bounded correction)
        z_repaired = z_asr_text + delta

    This allows the model to repair noisy CTC transcription embeddings
    guided by the acoustic signal.
    """

    def __init__(
        self,
        audio_dim: int,       # fusion_dim
        text_dim: int,        # fusion_dim
        hidden_dim: int,
        output_dim: int,      # fusion_dim
        dropout: float = 0.1,
        delta_scale: float = 0.3,
    ):
        super().__init__()
        self.delta_scale = delta_scale

        self.mlp = nn.Sequential(
            nn.Linear(audio_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        z_audio: torch.Tensor,  # [B, audio_dim]
        z_asr: torch.Tensor,    # [B, text_dim]
        alpha: Optional[torch.Tensor] = None, # [B, 1]
    ) -> torch.Tensor:
        """
        Returns:
            z_repaired: [B, fusion_dim]  — repaired ASR text embedding
        """
        x = torch.cat([z_audio, z_asr], dim=-1)   # [B, audio_dim + text_dim]
        delta_raw = self.mlp(x)                    # [B, output_dim]
        delta = self.delta_scale * torch.tanh(delta_raw)
        
        if alpha is not None:
            z_repaired = z_asr + alpha * delta
        else:
            z_repaired = z_asr + delta
            
        return z_repaired


class UncertaintyGate(nn.Module):
    """
    Uncertainty Gate: estimates confidence in ASR student transcription.

    Formula:
        alpha_raw = MLP([z_audio; z_asr])   -> scalar
        alpha     = clamp(sigmoid(alpha_raw), alpha_min, alpha_max)

    alpha ∈ (alpha_min, alpha_max):
        - High alpha → ASR text is reliable, use more of it
        - Low alpha  → ASR text is noisy, rely more on audio
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
    ):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.net = nn.Sequential(
            nn.Linear(audio_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_audio: torch.Tensor,  # [B, audio_dim]
        z_asr: torch.Tensor,    # [B, text_dim]
    ) -> torch.Tensor:
        """
        Returns:
            alpha: [B, 1]  — ASR reliability score in (alpha_min, alpha_max)
        """
        x = torch.cat([z_audio, z_asr], dim=-1)   # [B, audio_dim + text_dim]
        alpha_raw = self.net(x)                    # [B, 1]
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(alpha_raw)
        return alpha

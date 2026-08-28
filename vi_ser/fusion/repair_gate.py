"""
vi_ser/fusion/repair_gate.py

Repair Network for the Student Branch (Hybrid AURORA).
Uses ASR Confidence (Uncertainty) to weight between noisy ASR text features and ideal hallucinated text features.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UncertaintyGate(nn.Module):
    """
    Computes a confidence scalar alpha in [0, 1] from CTC logits based on Entropy.
    alpha ~ 1.0 -> High confidence (ASR is clear)
    alpha ~ 0.0 -> Low confidence (ASR is noisy/uncertain)
    """
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))
        
        # We use raw entropy-based confidence directly instead of a learnable projection
        # This prevents random initializations from allowing noisy ASR text to leak early in training.

    def forward(self, logits_ctc: torch.Tensor, audio_mask: torch.Tensor = None) -> torch.Tensor:
        """
        logits_ctc: [B, T, V]
        audio_mask: [B, T] (True = valid, False = pad)
        Returns:
            alpha: [B, 1]
        """
        probs = F.softmax(logits_ctc, dim=-1) # [B, T, V]
        # Entropy H = - sum(p * log p)
        log_probs = torch.log(probs + 1e-9)
        entropy_frames = -torch.sum(probs * log_probs, dim=-1) # [B, T]
        
        if audio_mask is not None:
            # Mask out padding frames
            entropy_frames = entropy_frames * audio_mask.float()
            # Mean entropy over valid frames
            seq_entropy = entropy_frames.sum(dim=1) / audio_mask.float().sum(dim=1).clamp(min=1) # [B]
        else:
            seq_entropy = entropy_frames.mean(dim=1) # [B]
            
        # Normalize to [0, 1]
        norm_entropy = seq_entropy / self.max_entropy.to(seq_entropy.device)
        norm_entropy = norm_entropy.clamp(0.0, 1.0)
        
        # Confidence raw = 1 - normalized_entropy
        confidence = 1.0 - norm_entropy # [B]
        
        # Use confidence directly as the gate value (already in [0, 1])
        alpha = confidence.unsqueeze(1) # [B, 1]
        
        return alpha


class RepairNetwork(nn.Module):
    """
    Merges Noisy ASR Text Features with Hallucinated Clean Text Features
    based on the ASR Uncertainty Gate.
    """
    def __init__(self, fusion_dim: int, vocab_size: int = 32):
        super().__init__()
        self.uncertainty_gate = UncertaintyGate(vocab_size=vocab_size)
        
        # Refinement network to smooth the interpolated embedding
        self.refine = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self, 
        z_text_asr: torch.Tensor, 
        z_hallucinated: torch.Tensor, 
        logits_ctc: torch.Tensor,
        audio_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        z_text_asr: [B, fusion_dim] - Features from BERT(ASR Text)
        z_hallucinated: [B, fusion_dim] - Features from HallucinationMLP(Audio)
        logits_ctc: [B, T, V] - Output from Wav2Vec2 CTC head
        
        Returns:
            z_text_repaired: [B, fusion_dim]
        """
        # 1. Compute confidence score alpha [B, 1]
        alpha = self.uncertainty_gate(logits_ctc, audio_mask)
        
        # 2. Interpolate based on confidence
        # High confidence -> trust ASR text. Low confidence -> trust hallucinated text.
        z_interpolated = alpha * z_text_asr + (1 - alpha) * z_hallucinated
        
        # 3. Refine and add residual
        z_refined = self.refine(z_interpolated)
        z_text_repaired = self.norm(z_interpolated + z_refined)
        
        return z_text_repaired, alpha

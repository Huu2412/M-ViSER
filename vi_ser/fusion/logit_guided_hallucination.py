"""
vi_ser/fusion/logit_guided_hallucination.py

Logit-Guided Hallucination modules for the Student path.

Architecture:
    1. LogitGuidedAttentionPooling:
       Uses CTC logits (vocab_size-dim probability distribution) as attention weights
       to selectively amplify speech-rich frames in the audio hidden states.
       
    2. HallucinationMLP:
       Takes the ASR-aware audio features and "hallucinates" (imagines) a representation
       that approximates what the Teacher's Cross-Attention would produce.
       Trained via Knowledge Distillation loss (Cosine with teacher_rep).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LogitGuidedAttentionPooling(nn.Module):
    """
    Logit-Guided Attention Pooling.

    Uses CTC logits as a "phonetic highlighter" to weight audio frames.
    Frames where the ASR model confidently predicts speech characters 
    receive higher attention; silence/noise frames are suppressed.

    Input:
        hidden_states: [B, T, H]       — Wav2Vec2 frame-level hidden states
        logits_ctc:    [B, T, V]       — CTC logits (before softmax)
        audio_mask:    [B, T]          — Boolean mask (True=valid frame)

    Output:
        pooled:        [B, H]          — ASR-aware utterance embedding
    """

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        # Project CTC logits (V-dim) to a scalar attention weight per frame
        self.logit_proj = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, T, H]
        logits_ctc: torch.Tensor,      # [B, T, V]
        audio_mask: torch.Tensor,      # [B, T] bool
    ) -> torch.Tensor:
        """Returns: [B, H] — ASR-aware pooled audio embedding."""
        # Convert CTC logits to per-frame attention scores
        # softmax over vocab gives phonetic distribution; project to scalar weight
        ctc_probs = F.softmax(logits_ctc.float(), dim=-1)   # [B, T, V]
        attn_scores = self.logit_proj(ctc_probs).squeeze(-1) # [B, T]

        # Mask out padding frames with -inf before softmax
        attn_scores = attn_scores.masked_fill(~audio_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)        # [B, T]

        # Guard: if all frames are masked, softmax returns NaN → replace with uniform
        nan_mask = attn_weights.isnan().any(dim=-1)           # [B]
        if nan_mask.any():
            uniform = audio_mask.float() / audio_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            attn_weights[nan_mask] = uniform[nan_mask]

        attn_weights = self.dropout(attn_weights)

        # Weighted sum of hidden states
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # [B, H]

        return pooled


class HallucinationMLP(nn.Module):
    """
    Hallucination Network (Student path).

    Takes the ASR-aware audio embedding and "imagines" a fused representation
    that approximates the Teacher's Cross-Attention output.

    During training, KD loss forces this MLP to produce representations
    similar to the Teacher's z_teacher_rep (which has access to BERT clean text).
    This way, the Student internalizes textual semantics purely from audio signals.

    Input:
        z_asr_aware:  [B, H]     — from LogitGuidedAttentionPooling
        z_audio:      [B, D]     — mean-pooled audio projection (from acoustic_encoder)

    Output:
        z_student_rep: [B, fusion_dim] — hallucinated fused representation
    """

    def __init__(
        self,
        audio_hidden_size: int,
        fusion_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Input: concatenation of [z_asr_aware (H); z_audio (D)]
        input_dim = audio_hidden_size + fusion_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
        )

        # Residual projection (from z_audio which is already fusion_dim)
        self.residual_proj = nn.Linear(fusion_dim, fusion_dim)

    def forward(
        self,
        z_asr_aware: torch.Tensor,   # [B, H]  — ASR-aware pooled embedding
        z_audio: torch.Tensor,       # [B, D]  — mean-pooled audio projection
    ) -> torch.Tensor:
        """Returns: [B, fusion_dim] — hallucinated student representation."""
        x = torch.cat([z_asr_aware, z_audio], dim=-1)  # [B, H + D]
        hallucinated = self.net(x)                       # [B, fusion_dim]

        # Residual connection from z_audio (audio is always reliable)
        out = hallucinated + self.residual_proj(z_audio) # [B, fusion_dim]

        return out
"""
vi_ser/fusion/logit_guided_hallucination.py
"""

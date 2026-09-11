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
    """
    Fallback to standard nn.LayerNorm as PyTorch's eps already handles zero-variance.
    Random noise injection was removed to ensure reproducibility.
    """
    pass


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
        z_audio: torch.Tensor,       # [B, T_a, audio_input_dim]
        audio_mask: torch.Tensor,    # [B, T_a] (Boolean: True=valid, False=padding)
        z_text: torch.Tensor,        # [B, T_t, text_input_dim]
        text_mask: torch.Tensor,     # [B, T_t] (Boolean: True=valid, False=padding)
    ):
        # Project inputs to fusion_dim
        z_a_seq = self.audio_proj(z_audio)   # [B, T_a, fusion_dim]
        z_t_seq = self.text_proj(z_text)     # [B, T_t, fusion_dim]

        # Invert masks for nn.MultiheadAttention (True means ignore/pad)
        audio_pad_mask = ~audio_mask  # [B, T_a]
        text_pad_mask = ~text_mask    # [B, T_t]

        # Cross-Attention: Audio attends to Text (Query=Audio, Key/Value=Text)
        attn_a, _ = self.audio_cross_attn(
            query=z_a_seq, key=z_t_seq, value=z_t_seq,
            key_padding_mask=text_pad_mask
        )
        
        # Cross-Attention: Text attends to Audio (Query=Text, Key/Value=Audio)
        attn_t, _ = self.text_cross_attn(
            query=z_t_seq, key=z_a_seq, value=z_a_seq,
            key_padding_mask=audio_pad_mask
        )

        # Residual connection and LayerNorm
        z_audio_enc_seq = self.audio_norm(z_a_seq + attn_a)  # [B, T_a, fusion_dim]
        z_text_enc_seq = self.text_norm(z_t_seq + attn_t)    # [B, T_t, fusion_dim]

        # Pooling back to [B, fusion_dim]
        # Audio: Masked Mean Pooling
        audio_masked = z_audio_enc_seq * audio_mask.unsqueeze(-1).float()
        z_audio_enc = audio_masked.sum(dim=1) / audio_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        
        # Text: CLS Token extraction (BERT puts [CLS] at index 0)
        z_text_enc = z_text_enc_seq[:, 0, :]

        # Guard sau cross-modal projection
        if not torch.isfinite(z_audio_enc).all():
            logger.warning("NaN/Inf in z_audio_enc after CrossModalEncoder. Replacing with zeros.")
            z_audio_enc = torch.nan_to_num(z_audio_enc, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(z_text_enc).all():
            logger.warning("NaN/Inf in z_text_enc after CrossModalEncoder. Replacing with zeros.")
            z_text_enc = torch.nan_to_num(z_text_enc, nan=0.0, posinf=0.0, neginf=0.0)

        return z_audio_enc, z_text_enc


class AuroraCrossModalEncoders(nn.Module):
    """
    Faithful port of AURORA's CrossModalEncoders.

    Unlike CrossModalEncoders (which handles sequential [B, T, H] audio with masks),
    this class expects **pooled** inputs [B, D] — the same interface as AURORA's
    `_encode(text_feat, audio_feat)` method.

    Internally:
      1. Project text/audio to fusion_dim via Linear → ReLU → LayerNorm → Dropout
      2. Unsqueeze to [B, 1, fusion_dim] (treat each sample as a 1-token sequence)
      3. Bidirectional cross-attention: text attends to audio, audio attends to text
      4. Residual add (shared res_proj) + mean(dim=1) → [B, fusion_dim]

    Returns:
        z_text_enc:  [B, fusion_dim]
        z_audio_enc: [B, fusion_dim]
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        fusion_dim: int,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()

        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_input_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        self.cross_attention_text = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_audio = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Shared residual projection (mirrors AURORA's single res_proj)
        self.res_proj = nn.Linear(fusion_dim, fusion_dim)

    def forward(
        self,
        text_feat: torch.Tensor,   # [B, text_input_dim]  — pooled text (e.g. BERT CLS)
        audio_feat: torch.Tensor,  # [B, audio_input_dim] — pooled audio (e.g. mean-pooled Wav2Vec2)
    ):
        # Unsqueeze if already 2-D (AURORA does this explicitly)
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)   # [B, 1, text_input_dim]
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)  # [B, 1, audio_input_dim]

        text_encoded  = self.text_encoder(text_feat)    # [B, 1, fusion_dim]
        audio_encoded = self.audio_encoder(audio_feat)  # [B, 1, fusion_dim]

        # Bidirectional cross-attention (AURORA style)
        text_attn,  _ = self.cross_attention_text(text_encoded,  audio_encoded, audio_encoded)
        audio_attn, _ = self.cross_attention_audio(audio_encoded, text_encoded, text_encoded)

        # Residual add (shared projection as in AURORA)
        text_out  = self.res_proj(text_encoded)  + text_attn   # [B, 1, fusion_dim]
        audio_out = self.res_proj(audio_encoded) + audio_attn  # [B, 1, fusion_dim]

        # Pool back to [B, fusion_dim]  (AURORA: .mean(dim=1))
        z_text_enc  = text_out.mean(dim=1)   # [B, fusion_dim]
        z_audio_enc = audio_out.mean(dim=1)  # [B, fusion_dim]

        return z_text_enc, z_audio_enc

"""
vi_ser/fusion/aurora_teacher.py

Faithful implementation of the Teacher Architecture from AURORA:
Repository: https://github.com/nhut-ngnn/AURORA.git

Architecture Components:
1. CrossModalEncoders:
   - Text & Audio linear projection to fusion_dim (512) + ReLU + LayerNorm + Dropout(0.3)
   - Bidirectional Multi-Head Cross-Attention (num_heads=4, dropout=0.3)
   - Shared residual linear projection
   - Mean-pooling along sequence dimension -> z_clean_text, z_audio_teacher [B, fusion_dim]

2. AudioGuidedGatedFusion (GMU):
   - Linear projection for audio and text
   - Additive combination: f_at = f_a + f_t
   - Refinement FFN: Linear(512, 512) -> GELU -> Dropout(0.2) -> Linear(512, 512)
   - Audio-guided gate: w = sigmoid(gate_audio(f_a) + gate_fusion(f_at'))
   - Gated output: fused = out_proj((1 - w) * f_a + w * f_at') -> teacher_rep [B, fusion_dim]

3. AuroraMLPClassifier:
   - 2-layer MLP classifier with ReLU, LayerNorm, and Dropout:
     Linear(512, 512) -> ReLU -> LayerNorm -> Dropout(0.3) ->
     Linear(512, 128) -> ReLU -> LayerNorm -> Dropout(0.3) ->
     Linear(128, num_classes) -> logits_teacher [B, num_classes]
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class AuroraCrossModalEncoders(nn.Module):
    """
    Bidirectional Multi-Head Cross-Attention for Audio and Text (AURORA).
    Receives pooled [B, D] or sequential [B, 1, D] embeddings.
    """
    def __init__(
        self,
        text_input_dim: int = 768,
        audio_input_dim: int = 768,
        fusion_dim: int = 512,
        dropout: float = 0.3,
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

        self.res_proj = nn.Linear(fusion_dim, fusion_dim)

    def forward(
        self,
        text_feat: torch.Tensor,   # [B, text_input_dim] or [B, 1, text_input_dim]
        audio_feat: torch.Tensor,  # [B, audio_input_dim] or [B, 1, audio_input_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)

        text_encoded = self.text_encoder(text_feat)
        audio_encoded = self.audio_encoder(audio_feat)

        text_attn, _ = self.cross_attention_text(text_encoded, audio_encoded, audio_encoded)
        audio_attn, _ = self.cross_attention_audio(audio_encoded, text_encoded, text_encoded)

        text_out = self.res_proj(text_encoded) + text_attn
        audio_out = self.res_proj(audio_encoded) + audio_attn

        z_text = text_out.mean(dim=1)    # [B, fusion_dim]
        z_audio = audio_out.mean(dim=1)  # [B, fusion_dim]
        return z_text, z_audio


class AuroraAudioGuidedGMU(nn.Module):
    """
    Audio-Guided Gated Multimodal Unit (GMU) from AURORA.
    Dynamically fuses text and audio embeddings with an audio-guided gate.
    """
    def __init__(
        self,
        text_dim: int = 512,
        audio_dim: int = 512,
        fusion_dim: int = 512,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.gate_audio = nn.Linear(fusion_dim, fusion_dim)
        self.gate_fusion = nn.Linear(fusion_dim, fusion_dim)
        self.out_proj = nn.Linear(fusion_dim, fusion_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        f_t = self.text_proj(text_feat)
        f_a = self.audio_proj(audio_feat)

        f_at = f_a + f_t
        f_at_prime = self.fusion_ffn(f_at) + f_at

        gate_pre = self.gate_audio(f_a) + self.gate_fusion(f_at_prime)
        w = self.sigmoid(gate_pre)

        fused = (1.0 - w) * f_a + w * f_at_prime
        fused = self.out_proj(fused)
        return fused


class AuroraMLPClassifier(nn.Module):
    """
    AURORA MLP Classifier with LayerNorm and ReLU.
    layer_dims defaults to [512, 128].
    """
    def __init__(
        self,
        input_dim: int = 512,
        layer_dims: Optional[List[int]] = None,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        if layer_dims is None:
            layer_dims = [512, 128]

        layers = []
        in_dim = input_dim
        for out_dim in layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AuroraTeacher(nn.Module):
    """
    Complete Teacher Architecture from AURORA (https://github.com/nhut-ngnn/AURORA.git).

    Forward pass:
        1. Encodes clean text & audio via CrossModalEncoders
        2. Fuses representations via AudioGuidedGMU
        3. Classifies emotion via AuroraMLPClassifier

    Inputs:
        text_clean: [B, text_input_dim] (Clean transcript embedding from BERT)
        audio:      [B, audio_input_dim] (Audio embedding from Wav2Vec2)

    Returns dict with:
        logits_teacher:   [B, num_classes]
        teacher_rep:      [B, fusion_dim]
        z_clean_text:     [B, fusion_dim]
        z_audio_teacher:  [B, fusion_dim]
    """
    def __init__(
        self,
        text_input_dim: int = 768,
        audio_input_dim: int = 768,
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        classifier_layer_dims: Optional[List[int]] = None,
        num_classes: int = 4,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        self.encoders = AuroraCrossModalEncoders(
            text_input_dim=text_input_dim,
            audio_input_dim=audio_input_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

        self.gmu = AuroraAudioGuidedGMU(
            text_dim=fusion_dim,
            audio_dim=fusion_dim,
            fusion_dim=fusion_dim,
            dropout_p=0.2,
        )

        self.classifier = AuroraMLPClassifier(
            input_dim=fusion_dim,
            layer_dims=classifier_layer_dims or [512, 128],
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        text_clean: torch.Tensor,
        audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_clean, z_audio_t = self.encoders(text_clean, audio)
        rep = self.gmu(z_clean, z_audio_t)
        logits = self.classifier(rep)

        return {
            "logits_teacher": logits,
            "teacher_rep": rep,
            "z_clean_text": z_clean,
            "z_audio_teacher": z_audio_t,
        }

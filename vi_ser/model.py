"""
vi_ser/model.py

SER Model: Speech Emotion Recognition — Logit-Guided Hallucination Architecture
==================================================================================
Integrates MTL-SER (CTC student ASR) + AURORA (Teacher Cross-Modal Distillation)
with Logit-Guided Hallucination for the Student path.

Architecture Overview:
──────────────────────────────────────────────────────────────────────────────
  Raw Audio
      │
      ▼
  Wav2Vec2 Encoder (Partial Fine-Tuning: top N layers unfrozen)
      │
      ├── CTC Head → logits_ctc  [B, T, V]  (ASR auxiliary task, CTC Loss)
      ├── hidden_states [B, T, H]
      └── z_audio [B, fusion_dim] (mean-pooled + projected)

  ┌────────────────────────────────────────────────────────────────────┐
  │                    Student Path (End-to-End)                        │
  │                                                                      │
  │  LogitGuidedAttentionPooling(hidden_states, logits_ctc)            │
  │    → z_asr_aware [B, H]  (ASR-enriched audio, phonetically gated) │
  │                                                                      │
  │  HallucinationMLP([z_asr_aware; z_audio])                          │
  │    → z_student_rep [B, fusion_dim]  (hallucinated fused repr)      │
  │                                                                      │
  │  EmotionClassifier(z_student_rep)                                   │
  │    → logits_emotion_student [B, num_emotion_classes]                │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │                    Teacher Path (training only)                      │
  │                                                                      │
  │  Ground-truth text → BERT → z_clean_text                           │
  │  CrossModalEncoders(audio_hidden, z_clean_text)                    │
  │    → z_audio_enc, z_text_enc                                       │
  │  AudioGuidedGMU(z_audio_enc, z_text_enc, alpha=1.0)               │
  │    → z_teacher_rep [B, fusion_dim]                                 │
  │  TeacherEmotionHead(z_teacher_rep)                                  │
  │    → logits_emotion_teacher [B, num_emotion_classes]               │
  └────────────────────────────────────────────────────────────────────┘

  Knowledge Distillation:
    L_hallucination = 1 - cosine_sim(z_student_rep, z_teacher_rep)
    L_kd            = KL(student_logits || teacher_logits)
    L_distill       = MSE(z_student_rep, z_teacher_rep)
──────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import ViSERConfig
from .encoders.acoustic_encoder import Wav2Vec2AcousticEncoder
from .encoders.text_encoder import BERTTextEncoder
from .fusion.cross_modal import CrossModalEncoders
from .fusion.audio_guided_gmu import AudioGuidedGatedFusion
from .fusion.logit_guided_hallucination import LogitGuidedAttentionPooling, HallucinationMLP
from .fusion.classifiers import EmotionClassifier, TeacherEmotionHead


class SERModel(nn.Module):
    """
    Speech Emotion Recognition model — Logit-Guided Hallucination Architecture.

    Combines:
      - Wav2Vec2 acoustic backbone with CTC head (student ASR, from MTL-SER)
      - BERT text encoder (for teacher clean text only)
      - Student: LogitGuidedAttentionPooling + HallucinationMLP (End-to-End, no BERT)
      - Teacher: AURORA-style Cross-Attention + Audio-Guided GMU (training only)
      - Knowledge Distillation: Student learns to hallucinate Teacher's fused repr
    """

    def __init__(self, config: ViSERConfig):
        super().__init__()
        self.config = config

        # ── Acoustic Encoder (Wav2Vec2 + CTC head) ───────────────────────────
        self.acoustic_encoder = Wav2Vec2AcousticEncoder(config)

        # ── Text Encoder (BERT — used only by Teacher path) ──────────────────
        self.text_encoder = BERTTextEncoder(config)

        # ── Student Path: Logit-Guided Hallucination ─────────────────────────
        self.logit_attn_pool = LogitGuidedAttentionPooling(
            vocab_size=config.vocab_size,
            hidden_size=config.acoustic_hidden_size,
            dropout=config.dropout,
        )
        self.hallucination_mlp = HallucinationMLP(
            audio_hidden_size=config.acoustic_hidden_size,
            fusion_dim=config.fusion_dim,
            hidden_dim=getattr(config, "hallucination_hidden_dim", 512),
            dropout=config.dropout,
        )

        # ── Teacher Path: Cross-Modal Fusion (AURORA-style) ──────────────────
        self.teacher_cross_modal = CrossModalEncoders(
            audio_input_dim=config.acoustic_hidden_size,
            text_input_dim=config.text_hidden_size,
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
            num_heads=config.num_heads,
        )
        self.teacher_gmu = AudioGuidedGatedFusion(
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
        )

        # ── Classification Heads ─────────────────────────────────────────────
        self.emotion_classifier = EmotionClassifier(config)
        self.teacher_emotion_classifier = TeacherEmotionHead(config)

    def _student_forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        audio_mask: torch.Tensor,     # [B, T]
        logits_ctc: torch.Tensor,     # [B, T, V]
        z_audio: torch.Tensor,        # [B, fusion_dim]
    ) -> torch.Tensor:
        """
        Student path: Audio-only End-to-End.
        Uses CTC logits as attention guide, then hallucinates fused representation.

        Returns:
            z_student_rep: [B, fusion_dim]
        """
        # Step 1: Logit-Guided Attention Pooling
        #   CTC logits act as a phonetic highlighter on audio frames
        z_asr_aware = self.logit_attn_pool(hidden_states, logits_ctc, audio_mask)  # [B, H]

        # Step 2: Hallucination MLP
        #   Concatenates [z_asr_aware; z_audio] and hallucinates a fused repr
        z_student_rep = self.hallucination_mlp(z_asr_aware, z_audio)  # [B, fusion_dim]

        return z_student_rep

    def _teacher_forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        audio_mask: torch.Tensor,     # [B, T]
        teacher_texts: List[str],     # Ground-truth transcripts (clean)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher path (training only): Audio + Clean GT Text → teacher_rep + logits.
        Uses AURORA-style Cross-Attention + GMU with full confidence (alpha=1.0).

        Returns:
            z_teacher_rep:          [B, fusion_dim]
            logits_emotion_teacher: [B, num_emotion_classes]
        """
        # Encode clean text with BERT
        text_out = self.text_encoder(teacher_texts, device=hidden_states.device)
        text_hidden = text_out["hidden_states"]   # [B, T_t, text_hidden_size]
        text_mask = text_out["attention_mask"]     # [B, T_t]

        # Cross-modal alignment (Audio ↔ Clean Text)
        z_audio_enc, z_text_enc = self.teacher_cross_modal(
            hidden_states, audio_mask, text_hidden, text_mask
        )

        # Gated fusion with full confidence (teacher has clean text)
        alpha_ones = torch.ones(hidden_states.size(0), 1, device=hidden_states.device)
        z_teacher_rep = self.teacher_gmu(z_audio_enc, z_text_enc, alpha_ones)

        # Teacher emotion classification
        logits_emotion_teacher = self.teacher_emotion_classifier(z_teacher_rep)

        return z_teacher_rep, logits_emotion_teacher

    def forward(
        self,
        # ── Audio inputs ──────────────────────────────────────────────────────
        input_values: torch.Tensor,           # [B, T_audio]
        attention_mask: torch.Tensor = None,
        # ── Text inputs (Teacher only) ────────────────────────────────────────
        teacher_texts: List[str] = None,      # Ground-truth transcripts (training only)
        # ── Mode ──────────────────────────────────────────────────────────────
        training_mode: bool = True,           # True: teacher path enabled
        # ── Unused (kept for backward compat) ─────────────────────────────────
        student_texts: List[str] = None,      # No longer needed (End-to-End)
        processor=None,                       # No longer needed
    ) -> Dict:
        """
        Full forward pass.

        Returns dict with:
            logits_emotion_student: [B, num_emotion_classes]
            logits_ctc:             [B, T, vocab_size]
            z_student_rep:          [B, fusion_dim]
            z_audio:                [B, fusion_dim]
            hidden_states:          [B, T, H]
            --- teacher outputs (only if training_mode=True and teacher_texts provided) ---
            logits_emotion_teacher: [B, num_emotion_classes]
            z_teacher_rep:          [B, fusion_dim]
        """
        # ── Step 1: Acoustic Encoding (Wav2Vec2) ─────────────────────────────
        acoustic_out = self.acoustic_encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = acoustic_out["hidden_states"]  # [B, T, H]
        audio_mask    = acoustic_out["audio_mask"]      # [B, T]
        z_audio       = acoustic_out["z_audio"]         # [B, fusion_dim]
        logits_ctc    = acoustic_out["logits_ctc"]      # [B, T, V]

        # ── Step 2: Student Path (Logit-Guided Hallucination) ────────────────
        z_student_rep = self._student_forward(
            hidden_states, audio_mask, logits_ctc, z_audio
        )

        # ── Step 3: Emotion Classification (Student) ─────────────────────────
        logits_emotion_student = self.emotion_classifier(z_student_rep)

        output = {
            "logits_emotion_student": logits_emotion_student,
            "logits_ctc":             logits_ctc,
            "z_student_rep":          z_student_rep,
            "z_fused":                z_student_rep,      # backward compat alias
            "z_audio":                z_audio,
            "hidden_states":          hidden_states,
            "acoustic_encoder":       self.acoustic_encoder,
        }

        # ── Step 4: Teacher Path (training only) ─────────────────────────────
        if training_mode and teacher_texts is not None:
            z_teacher_rep, logits_emotion_teacher = self._teacher_forward(
                hidden_states, audio_mask, teacher_texts
            )
            output["z_teacher_rep"]          = z_teacher_rep
            output["logits_emotion_teacher"] = logits_emotion_teacher

        return output

    def freeze_acoustic_backbone(self):
        """Freeze all Wav2Vec2 parameters."""
        for param in self.acoustic_encoder.encoder.parameters():
            param.requires_grad = False

    def unfreeze_acoustic_backbone(self):
        """Unfreeze Wav2Vec2 for fine-tuning."""
        for param in self.acoustic_encoder.encoder.parameters():
            param.requires_grad = True
        self.acoustic_encoder._freeze_feature_extractor()

    def partial_unfreeze_acoustic(self, num_unfrozen_layers: int = None):
        """
        Partial Fine-Tuning: Unfreeze only the top N transformer layers + CTC head.
        This is the recommended approach for balancing compute and performance.
        
        Args:
            num_unfrozen_layers: Number of top transformer layers to unfreeze.
                                 If None, uses config.num_unfrozen_layers.
        """
        if num_unfrozen_layers is None:
            num_unfrozen_layers = getattr(self.config, "num_unfrozen_layers", 2)

        # First freeze everything
        self.freeze_acoustic_backbone()

        # Unfreeze top N transformer layers
        encoder_layers = self.acoustic_encoder.encoder.encoder.layers
        total_layers = len(encoder_layers)
        if num_unfrozen_layers > 0:
            for layer in encoder_layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Always unfreeze CTC head (needed for CTC loss)
        for param in self.acoustic_encoder.ctc_head.parameters():
            param.requires_grad = True

        # Always unfreeze audio projection (needed for z_audio)
        for param in self.acoustic_encoder.audio_proj.parameters():
            param.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per module."""
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


# Backward-compatible alias
ViSERModel = SERModel

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
    L_hallu         = Cosine(z_student_rep, z_teacher_rep)
──────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import ViSERConfig
from .encoders.acoustic_encoder import Wav2Vec2AcousticEncoder
from .encoders.text_encoder import BERTTextEncoder
from .fusion.cross_modal import CrossModalEncoders, AuroraCrossModalEncoders
from .fusion.audio_guided_gmu import AudioGuidedGatedFusion, AuroraGMU
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

        # ── Teacher Path: Cross-Modal Fusion (AURORA exact architecture) ─────
        # AuroraCrossModalEncoders: receives pooled [B, D] inputs, same as AURORA's
        # CrossModalEncoders — unsqueeze → cross-attn → mean(dim=1)
        # audio_input_dim = acoustic_hidden_size because z_audio_pooled is mean-pooled
        # directly from hidden_states [B, T, acoustic_hidden_size]
        self.teacher_cross_modal = AuroraCrossModalEncoders(
            text_input_dim=config.text_hidden_size,
            audio_input_dim=config.acoustic_hidden_size,  # mean-pooled from hidden_states
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
            num_heads=config.num_heads,
        )
        # AuroraGMU: forward(text_feat, audio_feat) — no alpha, exact AURORA logic
        self.teacher_gmu = AuroraGMU(
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
        Exact AURORA architecture: _encode(text_cls, z_audio) → gmu(z_clean, z_audio_t).

        Steps (mirrors AURORA's _forward_teacher):
          1. BERT → CLS token as pooled text embedding [B, text_hidden_size]
          2. Masked mean-pool audio hidden_states → z_audio_pooled [B, fusion_dim]
             (z_audio from acoustic_encoder is already at fusion_dim)
          3. AuroraCrossModalEncoders(text_cls, z_audio_pooled)
             → z_text_enc [B, fusion_dim], z_audio_enc [B, fusion_dim]
          4. AuroraGMU(z_text_enc, z_audio_enc) — no alpha (teacher has clean text)
             → z_teacher_rep [B, fusion_dim]
          5. TeacherEmotionHead(z_teacher_rep) → logits_emotion_teacher

        Returns:
            z_teacher_rep:          [B, fusion_dim]
            logits_emotion_teacher: [B, num_emotion_classes]
        """
        # Step 1: Encode clean text with BERT → extract CLS token [B, text_hidden_size]
        text_out = self.text_encoder(teacher_texts, device=hidden_states.device)
        text_hidden = text_out["hidden_states"]   # [B, T_t, text_hidden_size]
        text_cls = text_hidden[:, 0, :]           # [B, text_hidden_size]  (BERT CLS token)

        # Step 2: Mean-pool audio hidden_states with mask → [B, acoustic_hidden_size]
        # Then use z_audio (already at fusion_dim) from the acoustic encoder output
        # z_audio is passed via forward() and stored in output dict; here we derive it
        # from hidden_states using the same masked mean-pool as AURORA's _encode()
        audio_mask_float = audio_mask.float()  # [B, T]
        z_audio_pooled = (
            hidden_states * audio_mask_float.unsqueeze(-1)
        ).sum(dim=1) / audio_mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        # z_audio_pooled: [B, acoustic_hidden_size]
        # Project to fusion_dim to match AuroraCrossModalEncoders audio_input_dim
        # NOTE: AuroraCrossModalEncoders.audio_encoder handles the Linear projection

        # Step 3: Cross-modal alignment (AURORA's _encode equivalent)
        # AuroraCrossModalEncoders expects pooled [B, D] inputs
        z_text_enc, z_audio_enc = self.teacher_cross_modal(
            text_cls, z_audio_pooled
        )  # both [B, fusion_dim]

        # Step 4: Audio-Guided GMU — no alpha (AURORA teacher always has full confidence)
        z_teacher_rep = self.teacher_gmu(z_text_enc, z_audio_enc)  # [B, fusion_dim]

        # Step 5: Teacher emotion classification
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
        run_student: bool = True,             # Stage 2 or End-to-End
        run_teacher: bool = True,             # Stage 1 or End-to-End
        teacher_force_no_grad: bool = False,  # True in Stage 2
        # ── Unused (kept for backward compat) ─────────────────────────────────
        student_texts: List[str] = None,      # No longer needed (End-to-End)
        processor=None,                       # CTC tokenizer for ASR decoding in Student path
        training_mode: bool = None,           # Deprecated
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
        z_student_rep = None
        logits_emotion_student = None
        
        if run_student:
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
        }

        # ── Step 4: Teacher Path (training only) ─────────────────────────────
        if run_teacher and teacher_texts is not None:
            if teacher_force_no_grad:
                with torch.no_grad():
                    z_teacher_rep, logits_emotion_teacher = self._teacher_forward(
                        hidden_states, audio_mask, teacher_texts
                    )
            else:
                z_teacher_rep, logits_emotion_teacher = self._teacher_forward(
                    hidden_states, audio_mask, teacher_texts
                )
            output["z_teacher_rep"]          = z_teacher_rep
            output["logits_emotion_teacher"] = logits_emotion_teacher

        return output

    def freeze_teacher(self):
        """Freeze all teacher path components (used in Stage 2)."""
        if hasattr(self, "teacher_cross_modal"):
            for param in self.teacher_cross_modal.parameters():
                param.requires_grad = False
        if hasattr(self, "teacher_gmu"):
            for param in self.teacher_gmu.parameters():
                param.requires_grad = False
        if hasattr(self, "teacher_emotion_classifier"):
            for param in self.teacher_emotion_classifier.parameters():
                param.requires_grad = False
        if hasattr(self, "text_encoder"):
            for param in self.text_encoder.parameters():
                param.requires_grad = False

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

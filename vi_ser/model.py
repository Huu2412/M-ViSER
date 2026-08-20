"""
vi_ser/model.py

SER Model: Speech Emotion Recognition
======================================================
Integrates MTL-SER (CTC student ASR) + AURORA (Audio-Guided Repair Fusion).

Architecture Overview:
──────────────────────────────────────────────────────────────────────────────
  Raw Audio
      │
      ▼
  Wav2Vec2 Encoder ─────────────────────────────────────────────────────────┐
      │                                                                       │
      ├── CTC Head → logits_ctc  (Student ASR auxiliary, from MTL-SER)       │
      └── z_audio [B, fusion_dim] (mean-pooled + projected)                  │
                                                                              │
  CTC decode → text (student)                                                │
      │                                                                       │
  BERT → z_asr_student [B, fusion_dim]                                       │
                                                                              │
  ┌────────────────────────────────────────────────────────────────────┐     │
  │                    Student Path                                      │     │
  │  CrossModalEncoders(z_audio, z_asr_student)                        │     │
  │  → RepairMLP → z_repaired                                          │     │
  │  → UncertaintyGate → alpha                                         │     │
  │  → AudioGuidedGMU → z_fused                                        │     │
  └────────────────────────────────────────────────────────────────────┘     │
                                                                              │
  ┌────────────────────────────────────────────────────────────────────┐     │
  │                    Teacher Path (training only)                      │     │
  │  Ground-truth text → BERT → z_clean_text                           │     │
  │  CrossModalEncoders(z_audio, z_clean_text)                         │     │
  │  AudioGuidedGMU(alpha=1.0) → z_teacher_rep                         │     │
  │  TeacherEmotionHead → logits_emotion_teacher                       │     │
  └────────────────────────────────────────────────────────────────────┘     │
                                                                              │
  Classifiers:
      z_fused → EmotionClassifier → logits_emotion_student  (primary)
──────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import ViSERConfig
from .encoders.acoustic_encoder import Wav2Vec2AcousticEncoder
from .encoders.text_encoder import BERTTextEncoder
from .fusion.cross_modal import CrossModalEncoders
from .fusion.repair_gate import RepairMLP, UncertaintyGate
from .fusion.audio_guided_gmu import AudioGuidedGatedFusion
from .fusion.classifiers import EmotionClassifier, TeacherEmotionHead


class SERModel(nn.Module):
    """
    Speech Emotion Recognition model.

    Combines:
      - Wav2Vec2 acoustic backbone with CTC head (student ASR, from MTL-SER)
      - BERT text encoder (for CTC text and teacher clean text)
      - AURORA-style Audio-Guided Repair + Gated Fusion
      - Primary: Emotion classification
      - Auxiliary: CTC speech recognition
      - Teacher-student KD: Clean GT text teacher → CTC student
    """

    def __init__(self, config: ViSERConfig):
        super().__init__()
        self.config = config

        # ── Acoustic Encoder (Wav2Vec2 + CTC head) ───────────────────────────
        self.acoustic_encoder = Wav2Vec2AcousticEncoder(config)

        # ── Text Encoder (BERT) ──────────────────────────────────────────────
        self.text_encoder = BERTTextEncoder(config)

        # ── Shared Fusion Modules (AURORA-style) ─────────────────────────────
        # CrossModal: project text and audio sequences and apply bidirectional cross-attention
        self.shared_cross_modal = CrossModalEncoders(
            audio_input_dim=config.acoustic_hidden_size,
            text_input_dim=config.text_hidden_size,
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
            num_heads=config.num_heads,
        )
        self.repair_mlp = RepairMLP(
            audio_dim=config.fusion_dim,
            text_dim=config.fusion_dim,
            hidden_dim=config.repair_hidden_dim,
            output_dim=config.fusion_dim,
            dropout=config.dropout,
            delta_scale=config.delta_scale,
        )
        self.uncertainty_gate = UncertaintyGate(
            audio_dim=config.fusion_dim,
            text_dim=config.fusion_dim,
            hidden_dim=config.repair_hidden_dim,
            dropout=config.dropout,
            alpha_min=config.uncertainty_alpha_min,
            alpha_max=config.uncertainty_alpha_max,
        )
        self.shared_gmu = AudioGuidedGatedFusion(
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
        )

        self.emotion_classifier = EmotionClassifier(config)
        self.teacher_emotion_classifier = TeacherEmotionHead(config)

    def _student_forward(
        self,
        audio_hidden: torch.Tensor,  # [B, T_a, audio_hidden_size]
        audio_mask: torch.Tensor,    # [B, T_a]
        student_texts: List[str],    # CTC-decoded text
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Student path: audio + CTC text → z_fused via Repair Gate + GMU.

        Returns:
            z_fused:    [B, fusion_dim]
            z_repaired: [B, fusion_dim]
            alpha:      [B, 1]
        """
        # Encode student CTC text with BERT
        text_out = self.text_encoder(student_texts, device=audio_hidden.device)
        text_hidden = text_out["hidden_states"]
        text_mask = text_out["attention_mask"]

        # Cross-modal alignment
        z_audio_enc, z_text_enc = self.shared_cross_modal(
            audio_hidden, audio_mask, text_hidden, text_mask
        )

        # Uncertainty gate: how reliable is the CTC text?
        alpha = self.uncertainty_gate(z_audio_enc, z_text_enc)

        # Repair noisy CTC text embedding using audio guidance
        if getattr(self.config, "repair_use_alpha", False):
            z_repaired = self.repair_mlp(z_audio_enc, z_text_enc, alpha)
        else:
            z_repaired = self.repair_mlp(z_audio_enc, z_text_enc)

        # Audio-guided gated fusion
        z_fused = self.shared_gmu(z_audio_enc, z_repaired, alpha)

        return z_fused, z_repaired, alpha

    def _teacher_forward(
        self,
        audio_hidden: torch.Tensor,  # [B, T_a, audio_hidden_size]
        audio_mask: torch.Tensor,    # [B, T_a]
        teacher_texts: List[str],    # Ground-truth transcripts (clean)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher path (training only): audio + clean GT text → teacher_rep + logits.
        alpha=1.0 (full confidence in clean text, no repair needed).

        Returns:
            z_teacher_rep:         [B, fusion_dim]
            logits_emotion_teacher: [B, num_emotion_classes]
        """
        text_out = self.text_encoder(teacher_texts, device=audio_hidden.device)
        text_hidden = text_out["hidden_states"]
        text_mask = text_out["attention_mask"]
        
        z_audio_enc, z_text_enc = self.shared_cross_modal(
            audio_hidden, audio_mask, text_hidden, text_mask
        )

        # Teacher uses full alpha=1.0 (clean text → maximum confidence)
        alpha_ones = torch.ones(audio_hidden.size(0), 1, device=audio_hidden.device)
        z_teacher_rep = self.shared_gmu(z_audio_enc, z_text_enc, alpha_ones)

        logits_emotion_teacher = self.teacher_emotion_classifier(z_teacher_rep)
        return z_teacher_rep, logits_emotion_teacher

    def decode_ctc(self, logits_ctc: torch.Tensor, processor) -> List[str]:
        """
        Greedy CTC decode to get student text.

        Args:
            logits_ctc: [B, T, V]
            processor: Wav2Vec2Processor / Wav2Vec2CTCTokenizer

        Returns:
            List of decoded strings
        """
        pred_ids = logits_ctc.argmax(dim=-1)  # [B, T]
        texts = processor.batch_decode(pred_ids.cpu())
        return texts

    def forward(
        self,
        # ── Audio inputs ──────────────────────────────────────────────────────
        input_values: torch.Tensor,           # [B, T_audio]
        attention_mask: torch.Tensor = None,
        # ── Text inputs ───────────────────────────────────────────────────────
        student_texts: List[str] = None,      # CTC decoded text (or pre-decoded)
        teacher_texts: List[str] = None,      # Ground-truth transcripts (training only)
        # ── Processor for CTC decode (if student_texts not pre-decoded) ───────
        processor=None,
        # ── Mode ──────────────────────────────────────────────────────────────
        training_mode: bool = True,           # True: teacher path enabled
    ) -> Dict:
        """
        Full forward pass.

        Returns dict with:
            logits_emotion_student: [B, num_emotion_classes]
            logits_ctc:             [B, T, vocab_size]
            z_fused:                [B, fusion_dim]
            alpha:                  [B, 1]
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
        audio_mask    = acoustic_out["audio_mask"]     # [B, T]
        z_audio       = acoustic_out["z_audio"]        # [B, fusion_dim]
        logits_ctc    = acoustic_out["logits_ctc"]     # [B, T, V]

        # ── Step 2: Decode CTC text (student) ───────────────────────────────
        if student_texts is None or (len(student_texts) > 0 and student_texts[0] is None):
            # Online CTC decode (slower; prefer pre-decoded for training)
            if processor is not None:
                student_texts = self.decode_ctc(logits_ctc, processor)
            else:
                # Fallback: empty strings (audio-only mode)
                student_texts = [""] * input_values.size(0)

        # ── Step 3: Student Path (AURORA Repair + GMU) ───────────────────────
        z_fused, z_repaired, alpha = self._student_forward(hidden_states, audio_mask, student_texts)

        # ── Step 4: Emotion Classification ───────────────────────────────────
        logits_emotion_student = self.emotion_classifier(z_fused)    # [B, num_emo]

        output = {
            "logits_emotion_student": logits_emotion_student,
            "logits_ctc":             logits_ctc,
            "z_fused":                z_fused,
            "z_repaired":             z_repaired,
            "alpha":                  alpha,
            "z_audio":                z_audio,
            "hidden_states":          hidden_states,
            # Expose acoustic encoder for CTC loss length computation
            "acoustic_encoder":       self.acoustic_encoder,
        }

        # ── Step 5: Teacher Path (training only) ─────────────────────────────
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
        # Re-freeze CNN feature extractor
        self.acoustic_encoder._freeze_feature_extractor()

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per module."""
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


# Backward-compatible alias
ViSERModel = SERModel

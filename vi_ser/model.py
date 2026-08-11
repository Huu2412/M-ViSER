"""
vi_ser/model.py

ViSER: Vietnamese Speech Emotion Recognition Model
======================================================
Integrates MTL-SER (CTC student ASR) + AURORA (Audio-Guided Repair Fusion)
for Vietnamese speech with regional accent auxiliary task.

Architecture Overview:
──────────────────────────────────────────────────────────────────────────────
  Raw Audio
      │
      ▼
  ViP-VL Encoder ──────────────────────────────────────────────┐
      │                                                          │
      ├── CTC Head → logits_ctc (Student ASR auxiliary)          │
      └── z_audio [B, fusion_dim] (mean-pooled + projected)     │
                                                                 │
  CTC decode → Vietnamese text (student)                        │
      │                                                          │
  PhoBERT → z_asr_student [B, fusion_dim]                       │
                                                                 │
  ┌────────────────────────────────────────────────────────┐    │
  │                   Student Path                          │    │
  │                                                         │    │
  │  CrossModalEncoders(z_audio, z_asr_student)            │    │
  │  → RepairMLP → z_repaired                              │    │
  │  → UncertaintyGate → alpha                             │    │
  │  → AudioGuidedGMU → z_fused                            │    │
  │                                                         │    │
  └────────────────────────────────────────────────────────┘    │
                                                                 │
  ┌────────────────────────────────────────────────────────┐    │
  │                   Teacher Path (training only)          │    │
  │                                                         │    │
  │  PhoWhisper text → PhoBERT → z_clean_text             │    │
  │  CrossModalEncoders(z_audio, z_clean_text)             │    │
  │  AudioGuidedGMU(g=1.0, alpha=1.0) → z_teacher_rep     │    │
  │  TeacherEmotionHead → logits_teacher                   │    │
  │                                                         │    │
  └────────────────────────────────────────────────────────┘    │
                                                                 │
  Classifiers:                                                   │
      z_fused  → EmotionClassifier  → logits_emotion (primary)  │
      z_audio  → RegionalClassifier → logits_regional (aux)  ───┘
──────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import ViSERConfig
from .encoders.acoustic_encoder import VipVlAcousticEncoder
from .encoders.text_encoder import PhoBERTTextEncoder
from .fusion.cross_modal import CrossModalEncoders
from .fusion.repair_gate import RepairMLP, UncertaintyGate
from .fusion.audio_guided_gmu import AudioGuidedGatedFusion
from .fusion.classifiers import EmotionClassifier, RegionalClassifier, TeacherEmotionHead


class ViSERModel(nn.Module):
    """
    ViSER: Vietnamese Speech Emotion Recognition.

    Combines:
      - ViP-VL acoustic backbone with CTC head (student ASR, from MTL-SER)
      - PhoBERT text encoder (for CTC text and teacher text)
      - AURORA-style Audio-Guided Repair + Gated Fusion
      - Primary: Emotion classification
      - Auxiliary: Regional accent recognition (Bắc/Trung/Nam)
      - Auxiliary: CTC speech recognition
      - Teacher-student KD: PhoWhisper teacher → CTC student
    """

    def __init__(self, config: ViSERConfig):
        super().__init__()
        self.config = config

        # ── Acoustic Encoder (ViP-VL + CTC head) ────────────────────────────
        self.acoustic_encoder = VipVlAcousticEncoder(config)

        # ── Text Encoder (PhoBERT) ───────────────────────────────────────────
        self.text_encoder = PhoBERTTextEncoder(config)

        # ── Student Path: AURORA Fusion Modules ──────────────────────────────
        # CrossModal: both inputs already at fusion_dim (projected by respective encoders)
        self.student_cross_modal = CrossModalEncoders(
            audio_input_dim=config.fusion_dim,
            text_input_dim=config.fusion_dim,
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
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
        self.student_gmu = AudioGuidedGatedFusion(
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
        )

        # ── Teacher Path: AURORA Fusion (clean text, no repair needed) ───────
        self.teacher_cross_modal = CrossModalEncoders(
            audio_input_dim=config.fusion_dim,
            text_input_dim=config.fusion_dim,
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
        )
        self.teacher_gmu = AudioGuidedGatedFusion(
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
        )

        # ── Classifier Heads ─────────────────────────────────────────────────
        self.emotion_classifier  = EmotionClassifier(config)
        self.regional_classifier = RegionalClassifier(config)
        self.teacher_head        = TeacherEmotionHead(config)

    def _student_forward(
        self,
        z_audio: torch.Tensor,       # [B, fusion_dim]
        student_texts: List[str],    # CTC-decoded Vietnamese text
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Student path: audio + CTC text → z_fused via Repair Gate + GMU.

        Returns:
            z_fused:    [B, fusion_dim]
            z_repaired: [B, fusion_dim]
            alpha:      [B, 1]
        """
        # Encode student ASR text with PhoBERT
        z_asr_student = self.text_encoder(student_texts, device=z_audio.device)

        # Cross-modal alignment (both already at fusion_dim, this refines)
        z_audio_enc, z_text_enc = self.student_cross_modal(z_audio, z_asr_student)

        # Repair noisy CTC text embedding using audio guidance
        z_repaired = self.repair_mlp(z_audio_enc, z_text_enc)

        # Uncertainty gate: how reliable is the CTC student text?
        alpha = self.uncertainty_gate(z_audio_enc, z_text_enc)

        # Audio-guided gated fusion
        z_fused = self.student_gmu(z_audio_enc, z_repaired, alpha)

        return z_fused, z_repaired, alpha

    def _teacher_forward(
        self,
        z_audio: torch.Tensor,       # [B, fusion_dim]
        teacher_texts: List[str],    # PhoWhisper-transcribed Vietnamese text
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher path (training only): audio + clean text → teacher_rep + logits.
        Teacher text from PhoWhisper is higher quality, no repair needed.
        alpha=1.0 (full confidence in clean text).

        Returns:
            z_teacher_rep:    [B, fusion_dim]
            logits_teacher:   [B, num_emotion_classes]
        """
        z_clean_text = self.text_encoder(teacher_texts, device=z_audio.device)
        z_audio_enc, z_text_enc = self.teacher_cross_modal(z_audio, z_clean_text)

        # Teacher uses full alpha=1.0 (clean text, maximum confidence)
        alpha_ones = torch.ones(z_audio.size(0), 1, device=z_audio.device)
        z_teacher_rep = self.teacher_gmu(z_audio_enc, z_text_enc, alpha_ones)

        logits_teacher = self.teacher_head(z_teacher_rep)
        return z_teacher_rep, logits_teacher

    def decode_ctc(self, logits_ctc: torch.Tensor, processor) -> List[str]:
        """
        Greedy CTC decode to get student text.
        Uses the Wav2Vec2 processor tokenizer (same as MTL-SER).

        Args:
            logits_ctc: [B, T, V]
            processor: Wav2Vec2Processor with Vietnamese tokenizer

        Returns:
            List of decoded Vietnamese strings
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
        teacher_texts: List[str] = None,      # PhoWhisper text (training only)
        # ── Processor for CTC decode (if student_texts not pre-decoded) ───────
        processor = None,
        # ── Mode ──────────────────────────────────────────────────────────────
        training_mode: bool = True,           # True: teacher path enabled
    ) -> Dict:
        """
        Full forward pass.

        Returns dict with:
            logits_emotion_student: [B, num_emotion_classes]
            logits_ctc:             [B, T, vocab_size]
            logits_regional:        [B, num_regional_classes]
            z_fused:                [B, fusion_dim]
            alpha:                  [B, 1]
            --- teacher outputs (only if training_mode=True and teacher_texts provided) ---
            logits_emotion_teacher: [B, num_emotion_classes]
            z_teacher_rep:          [B, fusion_dim]
        """
        # ── Step 1: Acoustic Encoding (ViP-VL) ──────────────────────────────
        acoustic_out = self.acoustic_encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = acoustic_out["hidden_states"]  # [B, T, H]
        z_audio       = acoustic_out["z_audio"]        # [B, fusion_dim]
        logits_ctc    = acoustic_out["logits_ctc"]     # [B, T, V]

        # ── Step 2: Decode CTC text (student) ───────────────────────────────
        if student_texts is None:
            # Online CTC decode (slower; prefer pre-decoded for training)
            if processor is not None:
                student_texts = self.decode_ctc(logits_ctc, processor)
            else:
                # Fallback: empty strings (audio-only mode)
                student_texts = [""] * input_values.size(0)

        # ── Step 3: Student Path (AURORA Repair + GMU) ───────────────────────
        z_fused, z_repaired, alpha = self._student_forward(z_audio, student_texts)

        # ── Step 4: Emotion & Regional Classification ─────────────────────────
        logits_emotion_student = self.emotion_classifier(z_fused)    # [B, num_emo]
        logits_regional        = self.regional_classifier(z_audio)   # [B, num_reg]

        output = {
            "logits_emotion_student": logits_emotion_student,
            "logits_ctc":             logits_ctc,
            "logits_regional":        logits_regional,
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
                z_audio, teacher_texts
            )
            output["z_teacher_rep"]          = z_teacher_rep
            output["logits_emotion_teacher"] = logits_emotion_teacher

        return output

    def freeze_acoustic_backbone(self):
        """Freeze all ViP-VL parameters."""
        for param in self.acoustic_encoder.encoder.parameters():
            param.requires_grad = False

    def unfreeze_acoustic_backbone(self):
        """Unfreeze ViP-VL for fine-tuning."""
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

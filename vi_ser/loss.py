"""
vi_ser/loss.py

ViSER Combined Loss Function.

L_total = L_emotion                     (CE — primary emotion classification)
        + alpha_ctc     * L_ctc         (CTC — student ASR auxiliary task)
        + alpha_kd      * L_kd          (KL  — knowledge distillation student||teacher)
        + alpha_distill * L_distill     (MSE — representation alignment z_fused||z_teacher_rep)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ViSERLoss(nn.Module):
    """
    Combined loss for Vietnamese Speech Emotion Recognition.

    Combines:
      1. Emotion CE (primary)
      2. CTC ASR loss (student auxiliary — from MTL-SER)
      3. KL knowledge distillation (student emotion logits || teacher emotion logits)
      4. MSE representation alignment (z_fused || z_teacher_rep)
    """

    def __init__(self, config):
        super().__init__()
        self.alpha_student_emotion = getattr(config, "alpha_student_emotion", 1.0)
        self.alpha_teacher_emotion = getattr(config, "alpha_teacher_emotion", 0.0)
        self.alpha_ctc      = config.alpha_ctc
        self.alpha_kd       = config.alpha_kd
        self.alpha_distill  = config.alpha_distill
        self.temperature    = config.kd_temperature
        self.ctc_zero_infinity = config.ctc_zero_infinity

        label_smoothing = getattr(config, "label_smoothing", 0.0)
        
        # Emotion CE Loss (Primary & Teacher)
        emotion_weights = getattr(config, "emotion_class_weights", None)
        if emotion_weights is not None:
            emotion_weights = torch.tensor(emotion_weights, dtype=torch.float)
        self.ce_loss = nn.CrossEntropyLoss(weight=emotion_weights, label_smoothing=label_smoothing)



        self.mse_loss = nn.MSELoss()
        self.kl_loss  = nn.KLDivLoss(reduction="batchmean")

    def _ctc_loss(
        self,
        logits_ctc: torch.Tensor,      # [B, T, V]
        ctc_labels: torch.Tensor,      # [B, L] — padded with -100
        input_values: torch.Tensor,    # [B, T_audio] — for length computation
        acoustic_encoder,              # for get_feat_extract_output_lengths
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """CTC loss — mirrors MTL-SER _ctc_loss exactly."""
        if ctc_labels is None:
            return torch.tensor(0.0, device=logits_ctc.device)

        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones(
                input_values.shape[0], input_values.shape[1],
                dtype=torch.long,
                device=input_values.device,
            )
        )
        max_input_len = input_values.shape[1]
        max_output_len = logits_ctc.shape[1]
        input_lengths = acoustic_encoder.get_feat_extract_output_lengths(
            attention_mask.sum(-1), max_input_len, max_output_len
        )

        labels_mask = ctc_labels >= 0
        target_lengths = labels_mask.sum(-1)

        # ── Guard: skip samples where input_len < target_len (CTC requirement) ──
        valid_mask = input_lengths >= target_lengths
        if not valid_mask.all():
            import logging
            logging.getLogger(__name__).warning(
                f"CTC: {(~valid_mask).sum().item()} samples bị loại do input_length < target_length."
            )
            # Sửa lại ctc_labels của các sample lỗi để vượt qua vòng kiểm tra của F.ctc_loss
            # (chúng ta sẽ xoá loss của chúng đi sau)
            ctc_labels = ctc_labels.clone()
            ctc_labels[~valid_mask] = -100
            ctc_labels[~valid_mask, 0] = 0  # dummy token
            
            # Tính lại
            labels_mask = ctc_labels >= 0
            target_lengths = labels_mask.sum(-1)

        flattened_targets = ctc_labels.masked_select(labels_mask)

        # Clamp log_probs để tránh -inf
        log_probs = F.log_softmax(logits_ctc.float(), dim=-1).transpose(0, 1)
        log_probs = log_probs.clamp(min=-100.0)

        with torch.backends.cudnn.flags(enabled=False):
            loss_all = F.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.pad_token_id,
                reduction="none",
                zero_infinity=self.zero_infinity,
            )
            # Chỉ lấy loss của các sample hợp lệ
            loss = (loss_all * valid_mask.float().to(loss_all.device)).sum() / (valid_mask.sum().float().to(loss_all.device) + 1e-8)

        # Final guard: nếu vẫn NaN/Inf thì trả 0
        if not torch.isfinite(loss):
            return torch.tensor(0.0, device=logits_ctc.device)
        return loss

    def _kd_loss(
        self,
        logits_student: torch.Tensor,  # [B, num_classes]
        logits_teacher: torch.Tensor,  # [B, num_classes]
    ) -> torch.Tensor:
        """
        KL divergence knowledge distillation.
        Soft targets from teacher with temperature scaling.
        """
        T = self.temperature
        log_p_s = F.log_softmax(logits_student / T, dim=-1)
        p_t     = F.softmax(logits_teacher / T, dim=-1)
        return self.kl_loss(log_p_s, p_t) * (T * T)

    def forward(
        self,
        # ── Student outputs ──────────────────────────────────────────────────
        logits_emotion_student: torch.Tensor,    # [B, num_emotion_classes]
        logits_ctc:             torch.Tensor,    # [B, T, vocab_size]
        z_fused:                torch.Tensor,    # [B, fusion_dim]
        # ── Labels ───────────────────────────────────────────────────────────
        emotion_labels:   torch.Tensor,          # [B]
        ctc_labels:       torch.Tensor = None,   # [B, L] padded with -100
        input_values:     torch.Tensor = None,   # [B, T_audio]
        attention_mask:   torch.Tensor = None,
        # ── Teacher outputs (optional) ────────────────────────────────────────
        logits_emotion_teacher: torch.Tensor = None,  # [B, num_classes]
        z_teacher_rep:          torch.Tensor = None,  # [B, fusion_dim]
        # ── Acoustic encoder reference (for CTC lengths) ─────────────────────
        acoustic_encoder = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Returns:
            loss: scalar tensor
            loss_dict: dict of individual loss values for logging
        """
        device = logits_emotion_student.device
        loss_dict = {}

        # ── 1. Primary: Emotion Classification (CE) ──────────────────────────
        if self.ce_loss.weight is not None and self.ce_loss.weight.device != device:
            self.ce_loss.weight = self.ce_loss.weight.to(device)

        # Cast logits sang float32 để tránh overflow khi dùng fp16/bf16
        logits_emotion_student = logits_emotion_student.float()

        l_emotion_student = self.ce_loss(logits_emotion_student, emotion_labels)
        # Guard NaN từ CE (có thể xảy ra nếu logits bị inf)
        if not torch.isfinite(l_emotion_student):
            l_emotion_student = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict["l_emotion_student"] = l_emotion_student.item()
        loss_dict["l_emotion"] = l_emotion_student.item()  # backward compat

        l_emotion_teacher = torch.tensor(0.0, device=device)
        if logits_emotion_teacher is not None and self.alpha_teacher_emotion > 0:
            l_emotion_teacher = self.ce_loss(logits_emotion_teacher.float(), emotion_labels)
            if not torch.isfinite(l_emotion_teacher):
                l_emotion_teacher = torch.tensor(0.0, device=device)
        loss_dict["l_emotion_teacher"] = l_emotion_teacher.item()

        # ── 2. Auxiliary: CTC Student ASR ────────────────────────────────────
        l_ctc = torch.tensor(0.0, device=device)
        if (
            ctc_labels is not None
            and logits_ctc is not None
            and acoustic_encoder is not None
            and self.alpha_ctc > 0
        ):
            l_ctc = self._ctc_loss(
                logits_ctc, ctc_labels, input_values, acoustic_encoder, attention_mask
            )
        loss_dict["l_ctc"] = l_ctc.item()

        # ── 3. Knowledge Distillation (KL) ───────────────────────────────────
        l_kd = torch.tensor(0.0, device=device)
        if logits_emotion_teacher is not None and self.alpha_kd > 0:
            l_kd = self._kd_loss(logits_emotion_student, logits_emotion_teacher.float().detach())
            if not torch.isfinite(l_kd):
                l_kd = torch.tensor(0.0, device=device)
        loss_dict["l_kd"] = l_kd.item()

        # ── 4. Representation Alignment (MSE) ────────────────────────────────
        l_distill = torch.tensor(0.0, device=device)
        if z_teacher_rep is not None and self.alpha_distill > 0:
            l_distill = self.mse_loss(z_fused, z_teacher_rep.detach())
            if not torch.isfinite(l_distill):
                l_distill = torch.tensor(0.0, device=device)
        loss_dict["l_distill"] = l_distill.item()

        # ── Total Loss ────────────────────────────────────────────────────────
        l_total = (
            self.alpha_student_emotion * l_emotion_student
            + self.alpha_teacher_emotion * l_emotion_teacher
            + self.alpha_ctc      * l_ctc
            + self.alpha_kd       * l_kd
            + self.alpha_distill  * l_distill
        )
        loss_dict["l_total"] = l_total.item()

        return l_total, loss_dict

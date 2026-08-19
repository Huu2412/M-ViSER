"""
vi_ser/encoders/text_encoder.py

BERT Text Encoder.
Encodes transcribed text (from student CTC or ground-truth teacher) into dense embeddings.
Uses bert-base-uncased (or any BERT-compatible model) via HuggingFace transformers.
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional

logger = logging.getLogger(__name__)


class SafeLayerNorm(nn.LayerNorm):
    """
    LayerNorm với guard chống zero-variance.
    Khi BERT encode chuỗi rỗng "", CLS embedding có thể gần all-zero → NaN trong LayerNorm.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=-1, keepdim=True)
        if (std < 1e-6).any():
            x = x + torch.randn_like(x) * 1e-6
        return super().forward(x)


class BERTTextEncoder(nn.Module):
    """
    Encodes text strings into fixed-size dense embeddings.
    Uses [CLS] token representation from BERT as the utterance embedding,
    then projects to fusion_dim for AURORA-style cross-modal fusion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.text_model_name,
            cache_dir=config.cache_dir,
        )
        self.bert = AutoModel.from_pretrained(
            config.text_model_name,
            cache_dir=config.cache_dir,
        )

        if config.freeze_text_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Project BERT [CLS] dim → fusion_dim
        # Dùng SafeLayerNorm để tránh NaN khi text rỗng → CLS all-zero
        self.proj = nn.Sequential(
            nn.Linear(config.text_hidden_size, config.fusion_dim),
            nn.GELU(),
            SafeLayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode a list of text strings.

        Args:
            texts: List[str] of length B
            device: Target device

        Returns:
            z_text: [B, fusion_dim]
        """
        if device is None:
            device = next(self.parameters()).device

        # Replace None or empty with a dummy token để tránh empty-sequence issues
        # "[UNK]" cho BERT vẫn sinh CLS embedding hợp lệ (không all-zero)
        texts = [t if (t and isinstance(t, str) and t.strip()) else "[UNK]" for t in texts]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        outputs = self.bert(**encoding)
        # Use [CLS] token embedding as sentence representation
        cls_emb = outputs.last_hidden_state[:, 0, :].float()  # [B, text_hidden_size]

        # Guard: NaN có thể xuất hiện từ BERT nếu input ids sai
        if not torch.isfinite(cls_emb).all():
            logger.warning(
                "NaN/Inf detected in BERT CLS embedding. Replacing with zeros."
            )
            cls_emb = torch.nan_to_num(cls_emb, nan=0.0, posinf=0.0, neginf=0.0)

        z_text = self.proj(cls_emb)  # [B, fusion_dim]

        # Final guard
        if not torch.isfinite(z_text).all():
            logger.warning(
                "NaN/Inf detected in z_text after projection. Replacing with zeros."
            )
            z_text = torch.nan_to_num(z_text, nan=0.0, posinf=0.0, neginf=0.0)

        return z_text

    def forward_from_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode from pre-tokenized inputs (for batched pre-processing).

        Returns:
            z_text: [B, fusion_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :].float()
        return self.proj(cls_emb)


class NullTextEncoder(nn.Module):
    """
    Fallback text encoder that returns zero embeddings.
    Used when no text is available (audio-only inference mode).
    """

    def __init__(self, fusion_dim: int):
        super().__init__()
        self.fusion_dim = fusion_dim

    def forward(self, texts: List[str], device=None) -> torch.Tensor:
        B = len(texts)
        dev = device or torch.device("cpu")
        return torch.zeros(B, self.fusion_dim, device=dev)

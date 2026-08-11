"""
vi_ser/text_encoder.py

PhoBERT Text Encoder.
Encodes Vietnamese text (from student CTC or teacher PhoWhisper) into dense embeddings.
Uses vinai/phobert-base via HuggingFace transformers.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Union


class PhoBERTTextEncoder(nn.Module):
    """
    Encodes Vietnamese text strings into fixed-size dense embeddings.
    Uses [CLS] token representation from PhoBERT as the utterance embedding.

    Then projects to fusion_dim for AURORA-style cross-modal fusion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.phobert_model_name,
            cache_dir=config.cache_dir,
        )
        self.bert = AutoModel.from_pretrained(
            config.phobert_model_name,
            cache_dir=config.cache_dir,
        )

        if config.freeze_phobert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Project PhoBERT [CLS] dim → fusion_dim
        self.proj = nn.Sequential(
            nn.Linear(config.text_hidden_size, config.fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode a list of Vietnamese text strings.

        Args:
            texts: List[str] of length B
            device: Target device

        Returns:
            z_text: [B, fusion_dim]
        """
        if device is None:
            device = next(self.parameters()).device

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        outputs = self.bert(**encoding)
        # Use [CLS] token embedding as sentence representation
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, text_hidden_size]
        z_text = self.proj(cls_emb)                   # [B, fusion_dim]
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
        cls_emb = outputs.last_hidden_state[:, 0, :]
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

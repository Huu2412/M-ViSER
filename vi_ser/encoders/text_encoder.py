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
    Fallback to standard nn.LayerNorm as PyTorch's eps already handles zero-variance.
    Random noise injection was removed to ensure reproducibility.
    """
    pass


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

        # Removed projection to fusion_dim; it will be done in CrossModalEncoders

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
            dict containing:
                "hidden_states": [B, T_text, text_hidden_size]
                "attention_mask": [B, T_text] (Boolean: True for valid, False for padding)
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
        # Full sequence hidden states
        hidden_states = outputs.last_hidden_state  # [B, T_text, text_hidden_size]
        attention_mask = encoding["attention_mask"].bool()  # [B, T_text]

        # Guard: NaN có thể xuất hiện từ BERT nếu input ids sai
        if not torch.isfinite(hidden_states).all():
            logger.warning(
                "NaN/Inf detected in BERT hidden_states. Replacing with zeros."
            )
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask
        }

    def forward_from_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode from pre-tokenized inputs (for batched pre-processing).

        Returns:
            dict containing:
                "hidden_states": [B, T_text, text_hidden_size]
                "attention_mask": [B, T_text]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state.float()
        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask.bool()
        }


class NullTextEncoder(nn.Module):
    """
    Fallback text encoder that returns zero embeddings.
    Used when no text is available (audio-only inference mode).
    """

    def __init__(self, config):
        super().__init__()
        self.text_hidden_size = config.text_hidden_size

    def forward(self, texts: List[str], device=None) -> dict:
        B = len(texts)
        dev = device or torch.device("cpu")
        # Dummy sequence of length 1
        hidden_states = torch.zeros(B, 1, self.text_hidden_size, device=dev)
        attention_mask = torch.ones(B, 1, dtype=torch.bool, device=dev)
        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask
        }

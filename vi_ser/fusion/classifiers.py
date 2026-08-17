"""
vi_ser/classifiers.py

Classification heads:
  - EmotionClassifier:  z_fused → emotion logits [B, num_emotion_classes]
  - TeacherEmotionHead: z_teacher_rep → teacher logits (for KD; frozen gradient)
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    2-layer MLP classifier with dropout.
    Used for both emotion and regional classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmotionClassifier(MLPClassifier):
    """
    Primary task: Emotion recognition (4 classes by default).
    Input: z_fused [B, fusion_dim]
    Output: logits [B, num_emotion_classes]
    """

    def __init__(self, config):
        super().__init__(
            input_dim=config.fusion_dim,
            hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_emotion_classes,
            dropout=config.dropout,
        )




class TeacherEmotionHead(MLPClassifier):
    """
    Teacher path emotion head.
    Applied to z_teacher_rep (from PhoWhisper+audio fusion via clean text).
    Used only during training for KD — gradients from KD loss flow through
    the student, NOT through this head (teacher is frozen).
    """

    def __init__(self, config):
        super().__init__(
            input_dim=config.fusion_dim,
            hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_emotion_classes,
            dropout=config.dropout,
        )

# vi_ser/fusion package
from .aurora_teacher import AuroraTeacher, AuroraCrossModalEncoders, AuroraAudioGuidedGMU, AuroraMLPClassifier
from .cross_modal import CrossModalEncoders
from .audio_guided_gmu import AudioGuidedGatedFusion, AuroraGMU
from .logit_guided_hallucination import LogitGuidedAttentionPooling, HallucinationMLP
from .classifiers import EmotionClassifier, TeacherEmotionHead, MLPClassifier

__all__ = [
    "AuroraTeacher",
    "AuroraCrossModalEncoders",
    "AuroraAudioGuidedGMU",
    "AuroraMLPClassifier",
    "CrossModalEncoders",
    "AudioGuidedGatedFusion",
    "AuroraGMU",
    "LogitGuidedAttentionPooling",
    "HallucinationMLP",
    "EmotionClassifier",
    "TeacherEmotionHead",
    "MLPClassifier",
]

"""
vi_ser/factory.py

Factory methods for instantiating models, losses, optimizers, and other components.
Centralizes the creation logic to keep training scripts clean.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from transformers import Wav2Vec2CTCTokenizer

from .model import ViSERModel
from .loss import ViSERLoss
from .encoders.acoustic_encoder import AcousticFeatureExtractor
from .encoders.asr_teacher import PhoWhisperTeacher, TeacherTextCache


def create_model(config, device: torch.device = None) -> ViSERModel:
    """Instantiate the ViSER model."""
    model = ViSERModel(config)
    if device is not None:
        model = model.to(device)
    return model


def create_loss(config) -> ViSERLoss:
    """Instantiate the combined loss function."""
    return ViSERLoss(config)


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer with discriminative learning rates."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "acoustic_encoder" in name or "text_encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = AdamW([
        {"params": backbone_params, "lr": getattr(config, "backbone_lr", config.learning_rate)},
        {"params": head_params, "lr": getattr(config, "head_lr", config.learning_rate)}
    ], weight_decay=config.weight_decay)
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config):
    """Create learning rate scheduler."""
    scheduler_type = getattr(config, "scheduler_type", "plateau").lower()
    
    if scheduler_type == "cosine":
        # Simplified cosine annealing without warmup for now (or T_max = num_epochs)
        return CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs, 
            eta_min=getattr(config, "scheduler_min_lr", 1e-6)
        )
    else:
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )


def create_acoustic_feature_extractor(config) -> AcousticFeatureExtractor:
    """Load ViP-VL acoustic feature extractor."""
    return AcousticFeatureExtractor.from_pretrained(
        config.acoustic_model_name, cache_dir=config.cache_dir
    )


def create_ctc_tokenizer(config) -> Wav2Vec2CTCTokenizer:
    """Load CTC tokenizer."""
    try:
        return Wav2Vec2CTCTokenizer.from_pretrained(
            config.acoustic_model_name,
            cache_dir=config.cache_dir,
            do_lower_case=True,
            word_delimiter_token="|",
        )
    except TypeError:
        # Xảy ra khi repo không có vocab.json
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"CTC Tokenizer config missing in '{config.acoustic_model_name}'. Falling back to 'nguyenvulebinh/wav2vec2-base-vietnamese-250h'.")
        return Wav2Vec2CTCTokenizer.from_pretrained(
            "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
            cache_dir=config.cache_dir,
            do_lower_case=True,
            word_delimiter_token="|",
        )


def create_teacher_components(config, device: torch.device = None):
    """
    Load PhoWhisper teacher model and text cache.
    Returns: (teacher_model, teacher_cache)
    """
    import os
    cache_path = os.path.join(config.cache_dir or "cache_vi_ser", "teacher_texts.pt")
    teacher_cache = TeacherTextCache(cache_path)
    
    phowhisper_teacher = PhoWhisperTeacher(
        model_name=config.phowhisper_model_name,
        cache_dir=config.cache_dir,
        device=str(device) if device is not None else None,
    )
    return phowhisper_teacher, teacher_cache

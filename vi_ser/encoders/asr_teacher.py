"""
vi_ser/asr_teacher.py

PhoWhisper Teacher ASR Module.
Transcribes audio to Vietnamese text using vinai/PhoWhisper-medium (frozen).
Used during training to provide high-quality transcriptions for the teacher path.

At inference time, the teacher is NOT used — only the student CTC path runs.
"""

import os
import logging
import torch
from typing import List, Optional, Union
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)

logger = logging.getLogger(__name__)


class PhoWhisperTeacher:
    """
    Frozen PhoWhisper ASR model.
    Wraps vinai/PhoWhisper-medium to provide high-quality Vietnamese transcriptions.

    Usage:
        teacher = PhoWhisperTeacher("vinai/PhoWhisper-medium")
        texts = teacher.transcribe(audio_list, sampling_rate=16000)
    """

    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-medium",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        language: str = "vietnamese",
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading PhoWhisper teacher: {model_name}")

        self.processor = WhisperProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device)

        # Teacher is ALWAYS frozen
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        logger.info("PhoWhisper teacher loaded and frozen.")

    @torch.no_grad()
    def transcribe(
        self,
        audio_arrays: List[torch.Tensor],
        sampling_rate: int = 16000,
    ) -> List[str]:
        """
        Transcribe a batch of audio waveforms to Vietnamese text.

        Args:
            audio_arrays: List of [T] tensors (raw waveforms at 16kHz)
            sampling_rate: Sample rate of input audio

        Returns:
            List of transcribed strings
        """
        texts = []
        for i in range(0, len(audio_arrays), self.batch_size):
            batch_audio = audio_arrays[i : i + self.batch_size]

            # Convert to numpy for processor
            batch_np = [
                a.squeeze().cpu().numpy() if torch.is_tensor(a) else a
                for a in batch_audio
            ]

            inputs = self.processor(
                batch_np,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            predicted_ids = self.model.generate(
                inputs.input_features,
                language=self.language,
                task="transcribe",
            )

            batch_texts = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            texts.extend(batch_texts)

        return texts

    @torch.no_grad()
    def get_encoder_hidden_states(
        self,
        audio_arrays: List[torch.Tensor],
        sampling_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Return Whisper encoder hidden states for use as teacher representations.
        These are used for L_distill (MSE alignment with student z_fused).

        Returns:
            encoder_hidden: [B, T_enc, D_whisper]
        """
        batch_np = [
            a.squeeze().cpu().numpy() if torch.is_tensor(a) else a
            for a in audio_arrays
        ]

        inputs = self.processor(
            batch_np,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        encoder_out = self.model.model.encoder(inputs.input_features)
        return encoder_out.last_hidden_state  # [B, T_enc, D_whisper]


class TeacherTextCache:
    """
    Cache PhoWhisper transcriptions to disk to avoid re-running every epoch.
    Keys are audio file paths.
    """

    def __init__(self, cache_path: str = "cache_vi_ser/teacher_texts.pt"):
        self.cache_path = cache_path
        self._cache = {}
        if os.path.exists(cache_path):
            self._cache = torch.load(cache_path, weights_only=False)
            logger.info(
                f"Loaded {len(self._cache)} cached teacher transcriptions from {cache_path}"
            )

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key, None)

    def set(self, key: str, value: str):
        self._cache[key] = value

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(self._cache, self.cache_path)
        logger.info(f"Saved {len(self._cache)} teacher transcriptions to {self.cache_path}")

    def __len__(self):
        return len(self._cache)

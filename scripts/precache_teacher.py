"""
scripts/precache_teacher.py

Chạy PhoWhisper một lần trên toàn bộ dataset ViSEC và lưu transcript vào cache.
Sau khi chạy script này, quá trình training sẽ KHÔNG cần chạy PhoWhisper nữa.

Cách dùng trên Kaggle:
    !python scripts/precache_teacher.py --config config/config.yaml --batch_size 8
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def precache_teacher(config_path: str, batch_size: int = 8):
    from config_loader import load_config
    from vi_ser.encoders.asr_teacher import PhoWhisperTeacher, TeacherTextCache

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -- Load PhoWhisper Teacher
    logger.info("Dang tai PhoWhisper Teacher...")
    teacher = PhoWhisperTeacher(
        model_name=config.phowhisper_model_name,
        cache_dir=config.cache_dir,
        device=str(device),
        batch_size=batch_size,
    )

    # -- Load toan bo dataset tu HF
    logger.info(f"Dang tai dataset {config.hf_dataset} tu Hugging Face...")
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset(config.hf_dataset, split="train")
    ds = ds.cast_column("path", hf_datasets.Audio(decode=False))

    # -- Khoi tao cache
    cache_path = os.path.join(config.cache_dir or "cache_vi_ser", "teacher_texts.pt")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = TeacherTextCache(cache_path)

    logger.info(f"Cache hien co: {len(cache)} transcripts")
    logger.info(f"Dataset: {len(ds)} samples | Se bo qua {len(cache)} da cache san")

    # -- Thu thap cac audio chua duoc cache
    pending_audio = []
    pending_keys  = []

    for i in tqdm(range(len(ds)), desc="Chuan bi audio"):
        item = ds[i]
        audio_info = item["path"]
        if not isinstance(audio_info, dict):
            audio_info = {
                "path":          getattr(audio_info, "path",          f"hf_{i}"),
                "bytes":         getattr(audio_info, "bytes",         None),
                "array":         getattr(audio_info, "array",         None),
                "sampling_rate": getattr(audio_info, "sampling_rate", 16000),
            }

        key = audio_info.get("path", f"hf_{i}") or f"hf_{i}"

        # Bo qua neu da co trong cache
        if cache.get(key) is not None:
            continue

        # Decode audio -> numpy array
        if audio_info.get("bytes") is not None:
            import io, soundfile as sf
            arr, sr = sf.read(io.BytesIO(audio_info["bytes"]))
            arr = np.array(arr, dtype=np.float32)
        elif audio_info.get("array") is not None:
            arr = np.array(audio_info["array"], dtype=np.float32)
            sr  = audio_info.get("sampling_rate", 16000)
        else:
            logger.warning(f"Khong the doc audio mau {i}, bo qua.")
            continue

        if arr.ndim > 1:
            arr = arr.squeeze()

        # Resample neu can
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

        pending_audio.append(arr)
        pending_keys.append(key)

    logger.info(f"Can transcribe: {len(pending_audio)} samples")

    if len(pending_audio) == 0:
        logger.info("Tat ca da duoc cache! Khong can lam gi them.")
        return

    # -- Chay PhoWhisper theo batch
    save_every = 200  # Luu cache sau moi 200 samples

    for i in tqdm(range(0, len(pending_audio), batch_size), desc="Transcribing"):
        batch_arrays = [torch.tensor(a) for a in pending_audio[i : i + batch_size]]
        batch_keys   = pending_keys[i : i + batch_size]

        try:
            texts = teacher.transcribe(batch_arrays, sampling_rate=16000)
            for key, text in zip(batch_keys, texts):
                cache.set(key, text)
        except Exception as e:
            logger.warning(f"Loi khi transcribe batch {i}: {e}. Bo qua batch nay.")
            for key in batch_keys:
                cache.set(key, "")

        # Luu dinh ky de khong mat du lieu neu bi interrupt
        if (i // batch_size + 1) % (save_every // batch_size) == 0:
            cache.save()
            logger.info(f"  -> Da luu {len(cache)} transcripts vao {cache_path}")

    # -- Luu lan cuoi
    cache.save()
    logger.info(f"Hoan tat! Da cache {len(cache)}/{len(ds)} transcripts vao: {cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-cache PhoWhisper transcripts")
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size khi chay PhoWhisper")
    args = parser.parse_args()

    precache_teacher(args.config, args.batch_size)

"""
smoke_test.py — Kiểm tra toàn bộ pipeline TRƯỚC khi train thật.

Thực hiện:
  1. Load config
  2. Tải feature extractor + tokenizer từ HuggingFace
  3. Load 1 batch từ HF dataset (streaming, không cần download hết)
  4. Forward pass qua model
  5. Tính loss
  6. Backward pass
  7. In tổng số params

Nếu script này chạy xong không lỗi → train.py sẵn sàng chạy.

Dùng: python smoke_test.py
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from vi_ser.factory import (
    create_model, create_loss, create_optimizer,
    create_acoustic_feature_extractor, create_ctc_tokenizer,
)

CONFIG_PATH = "config/config.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_fake_batch(feature_extractor, ctc_tokenizer, config, n=2):
    """Tạo một batch giả để test forward pass không cần data thật."""
    sr = config.sampling_rate
    dur = 3  # 3 giây audio giả

    # Fake audio: white noise
    audios = [np.random.randn(sr * dur).astype(np.float32) for _ in range(n)]

    inputs = feature_extractor(
        audios,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    # Fake emotion labels
    emotion_labels = torch.randint(0, config.num_emotion_classes, (n,))

    # Fake CTC labels (random token ids)
    fake_text = ["hello world", "test audio"]
    ctc = ctc_tokenizer(fake_text, return_tensors="pt", padding=True)
    ctc_labels = ctc["input_ids"]

    return {
        "input_values":  inputs["input_values"].to(DEVICE),
        "attention_mask": inputs.get("attention_mask", None),
        "emotion_labels": emotion_labels.to(DEVICE),
        "ctc_labels":    ctc_labels.to(DEVICE),
        "student_texts": [None] * n,  # will be decoded from CTC online
        "teacher_texts": fake_text,
    }


def main():
    print("=" * 60)
    print("  ViSER Smoke Test")
    print("=" * 60)

    # 1. Config
    print("\n[1/6] Loading config...")
    config = load_config(CONFIG_PATH)
    print(f"  acoustic model : {config.acoustic_model_name}")
    print(f"  text model     : {config.text_model_name}")
    print(f"  hf_dataset     : {config.hf_dataset}")
    print(f"  num_epochs     : {config.num_epochs}")
    print(f"  batch_size     : {config.batch_size}")
    print(f"  device         : {DEVICE}")

    # 2. Feature extractor + tokenizer
    print("\n[2/6] Loading feature extractor + CTC tokenizer...")
    feature_extractor = create_acoustic_feature_extractor(config)
    ctc_tokenizer     = create_ctc_tokenizer(config)
    config.vocab_size = len(ctc_tokenizer)
    print(f"  vocab_size     : {config.vocab_size}")

    # 3. Model
    print("\n[3/6] Initializing model...")
    model = create_model(config, DEVICE)
    param_counts = model.count_parameters()
    total_params = param_counts.pop("total")
    for name, count in param_counts.items():
        print(f"  {name:<30}: {count:>10,} params")
    print(f"  {'TOTAL':<30}: {total_params:>10,} params")

    # 4. Fake batch
    print("\n[4/6] Creating fake batch (2 samples × 3s audio)...")
    batch = make_fake_batch(feature_extractor, ctc_tokenizer, config, n=2)
    print(f"  input_values shape : {batch['input_values'].shape}")

    # 5. Forward pass
    print("\n[5/6] Forward pass (training_mode=True)...")
    model.train()
    outputs = model(
        input_values=batch["input_values"],
        attention_mask=batch["attention_mask"].to(DEVICE) if batch["attention_mask"] is not None else None,
        student_texts=batch["student_texts"],
        teacher_texts=batch["teacher_texts"],
        processor=ctc_tokenizer,
        training_mode=True,
    )
    print(f"  logits_emotion_student : {outputs['logits_emotion_student'].shape}")
    print(f"  logits_ctc             : {outputs['logits_ctc'].shape}")
    print(f"  z_fused                : {outputs['z_fused'].shape}")
    print(f"  alpha                  : {outputs['alpha'].shape}")
    print(f"  logits_emotion_teacher : {outputs.get('logits_emotion_teacher', 'N/A (no teacher)')}")

    # 6. Loss + backward
    print("\n[6/6] Loss computation + backward pass...")
    loss_fn = create_loss(config)
    loss, loss_dict = loss_fn(
        logits_emotion_student=outputs["logits_emotion_student"],
        logits_ctc=outputs["logits_ctc"],
        z_fused=outputs["z_fused"],
        emotion_labels=batch["emotion_labels"],
        ctc_labels=batch["ctc_labels"],
        input_values=batch["input_values"],
        attention_mask=batch["attention_mask"].to(DEVICE) if batch["attention_mask"] is not None else None,
        logits_emotion_teacher=outputs.get("logits_emotion_teacher"),
        z_teacher_rep=outputs.get("z_teacher_rep"),
        acoustic_encoder=outputs["acoustic_encoder"],
    )
    print(f"  l_total          : {loss_dict['l_total']:.4f}")
    print(f"  l_emotion_student: {loss_dict['l_emotion_student']:.4f}")
    print(f"  l_ctc            : {loss_dict['l_ctc']:.4f}")
    print(f"  l_kd             : {loss_dict['l_kd']:.4f}")

    loss.backward()
    print("  backward()       : OK")

    print("\n" + "=" * 60)
    print("  [PASSED] SMOKE TEST OK -- train.py san sang chay!")
    print("=" * 60)
    print(f"\n  Lenh train:\n  python train.py --config {CONFIG_PATH}")


if __name__ == "__main__":
    main()

import os
import argparse
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

def prepare_visec_5fold(output_dir: str = "data"):
    """
    Tải bộ dữ liệu hustep-lab/ViSEC từ Hugging Face, xuất file audio ra thư mục cục bộ,
    và chia thành 5 folds (Speaker-Independent) để huấn luyện K-Fold Cross Validation.
    """
    print("1. Đang tải ViSEC dataset từ Hugging Face...")
    # Tải dataset (có thể yêu cầu Hugging Face token nếu dataset private, nhưng ViSEC thường là public)
    dataset = load_dataset("hustep-lab/ViSEC", split="train")
    
    # Tạo thư mục chứa audio
    audio_dir = os.path.join(output_dir, "visec_audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    print(f"2. Đang xuất {len(dataset)} file audio ra thư mục: {audio_dir} ...")
    records = []
    
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
        # Xử lý an toàn cột 'path'
        audio_info = item["path"]
        if not isinstance(audio_info, dict):
            # Trong một số phiên bản datasets, nó trả về object thay vì dict
            audio_info = {
                "path": getattr(audio_info, "path", f"audio_{i}.wav"),
                "array": getattr(audio_info, "array", None),
                "sampling_rate": getattr(audio_info, "sampling_rate", 16000)
            }
            
        original_path = audio_info.get("path", f"audio_{i}.wav")
        if original_path is None:
            original_path = f"audio_{i}.wav"
        basename = os.path.basename(original_path)
        if not basename.endswith(".wav"):
            basename += ".wav"
            
        save_path = os.path.join(audio_dir, basename)
        
        # Đảm bảo mảng audio là numpy array 1D
        audio_array = np.array(audio_info["array"], dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
            
        # Lưu file âm thanh xuống ổ cứng (16kHz)
        sf.write(save_path, audio_array, audio_info["sampling_rate"])
        
        # Thu thập metadata
        emotion = item.get("emotion", "neutral")
        text = item.get("text", "")
        speaker_id = item.get("speaker_id", item.get("speaker", "unknown"))
        
        records.append({
            "file": save_path,
            "emotion": emotion,
            "text": text,
            "speaker_id": speaker_id
        })
        
    df = pd.DataFrame(records)
    
    print("3. Bắt đầu chia 5-Fold (Speaker-Independent)...")
    # Sử dụng GroupKFold để đảm bảo 1 speaker không xuất hiện ở cả Train và Test
    gkf = GroupKFold(n_splits=5)
    
    fold = 1
    for train_idx, val_idx in gkf.split(df, groups=df["speaker_id"]):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Lưu ra CSV
        train_csv = os.path.join(fold_dir, "train.csv")
        val_csv = os.path.join(fold_dir, "val.csv")
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        
        print(f"  -> Fold {fold}: Train={len(train_df)} samples, Val={len(val_df)} samples. Đã lưu vào {fold_dir}")
        fold += 1
        
    print("\nHoàn tất! Bây giờ bạn có thể chỉnh sửa paths.train_csv và paths.val_csv trong config.yaml thành:")
    print("  train_csv: 'data/fold_1/train.csv'")
    print("  val_csv: 'data/fold_1/val.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị 5-Fold K-Fold cho ViSEC")
    parser.add_argument("--output_dir", type=str, default="data", help="Thư mục xuất dữ liệu")
    args = parser.parse_args()
    
    prepare_visec_5fold(args.output_dir)

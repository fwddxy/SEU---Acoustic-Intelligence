"""使用训练好的 ESC-10 分类器识别一个音频文件。"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.extract_features import mfcc_feature, yamnet_embedding_feature
from src.yamnet import load_audio, predict_waveform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
LABEL_NAMES = (
    "dog",
    "rooster",
    "rain",
    "sea_waves",
    "crackling_fire",
    "crying_baby",
    "sneezing",
    "clock_tick",
    "helicopter",
    "chainsaw",
)
DISPLAY_LABELS = {
    "dog": "狗叫声",
    "rooster": "公鸡打鸣",
    "rain": "雨声",
    "sea_waves": "海浪声",
    "crackling_fire": "炉火噼啪声",
    "crying_baby": "婴儿哭声",
    "sneezing": "喷嚏声",
    "clock_tick": "时钟滴答声",
    "helicopter": "直升机声",
    "chainsaw": "电锯声",
}
MODEL_NAMES = ("mfcc_svm", "yamnet_svm", "yamnet_mlp")


@lru_cache(maxsize=3)
def _load_model(name: str):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown classifier: {name}")
    return joblib.load(MODEL_DIR / f"{name}.joblib")


def predict_features(
    waveform: np.ndarray, sample_rate: int, embeddings: np.ndarray, name: str
) -> list[dict]:
    model = _load_model(name)
    if name == "mfcc_svm":
        vector = mfcc_feature(waveform, sample_rate)
    elif getattr(model, "n_features_in_", 1024) == 3072:
        vector = yamnet_embedding_feature(embeddings)
    else:
        vector = embeddings.mean(axis=0).astype(np.float32)
    probabilities = model.predict_proba(vector.reshape(1, -1))[0]
    order = np.argsort(probabilities)[::-1][:3]
    return [
        {"label": LABEL_NAMES[int(model.classes_[index])], "score": float(probabilities[index])}
        for index in order
    ]


def predict_file(audio_path: Path) -> None:
    waveform, sample_rate = load_audio(audio_path)
    embeddings = predict_waveform(waveform)["embeddings"]
    for name in MODEL_NAMES:
        if not (MODEL_DIR / f"{name}.joblib").exists():
            print(f"{name}：模型文件不存在，请先运行 train.py")
            continue
        results = predict_features(waveform, sample_rate, embeddings, name)
        model_label = {
            "mfcc_svm": "MFCC + 支持向量机",
            "yamnet_svm": "YAMNet + 支持向量机",
            "yamnet_mlp": "YAMNet + 多层感知机",
        }[name]
        print(model_label)
        for rank, item in enumerate(results, start=1):
            label = DISPLAY_LABELS.get(item["label"], item["label"])
            print(f"  {rank}. {label}：{item['score']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="音频文件路径")
    args = parser.parse_args()
    if not args.audio.exists():
        raise SystemExit(f"找不到音频文件：{args.audio}")
    predict_file(args.audio)


if __name__ == "__main__":
    main()

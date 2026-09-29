"""优先使用 ESC-50 五十分类模型识别音频，保留旧十类模型兼容。"""

from __future__ import annotations

import argparse
import json
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
    "pig": "猪叫声",
    "cow": "牛叫声",
    "frog": "青蛙声",
    "cat": "猫叫声",
    "hen": "母鸡叫声",
    "insects": "昆虫声",
    "sheep": "羊叫声",
    "crow": "乌鸦叫声",
    "crickets": "蟋蟀声",
    "chirping_birds": "鸟鸣声",
    "water_drops": "水滴声",
    "wind": "风声",
    "pouring_water": "倒水声",
    "toilet_flush": "冲水声",
    "thunderstorm": "雷暴声",
    "clapping": "鼓掌声",
    "breathing": "呼吸声",
    "coughing": "咳嗽声",
    "footsteps": "脚步声",
    "laughing": "笑声",
    "brushing_teeth": "刷牙声",
    "snoring": "打鼾声",
    "drinking_sipping": "喝水声",
    "door_wood_knock": "敲木门声",
    "mouse_click": "鼠标点击声",
    "keyboard_typing": "键盘打字声",
    "door_wood_creaks": "木门吱呀声",
    "can_opening": "开罐声",
    "washing_machine": "洗衣机声",
    "vacuum_cleaner": "吸尘器声",
    "clock_alarm": "闹钟声",
    "glass_breaking": "玻璃破碎声",
    "siren": "警报声",
    "car_horn": "汽车喇叭声",
    "engine": "发动机声",
    "train": "火车声",
    "church_bells": "教堂钟声",
    "airplane": "飞机声",
    "fireworks": "烟花声",
    "hand_saw": "手锯声",
}
MODEL_NAMES = ("mfcc_svm", "yamnet_svm", "yamnet_mlp")
ESC50_LABELS_PATH = MODEL_DIR / "esc50_labels.json"


@lru_cache(maxsize=3)
def _load_model(name: str):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown classifier: {name}")
    full_model = MODEL_DIR / f"esc50_{name}.joblib"
    if full_model.exists() and ESC50_LABELS_PATH.exists():
        return joblib.load(full_model)
    return joblib.load(MODEL_DIR / f"{name}.joblib")


@lru_cache(maxsize=1)
def _esc50_label_names() -> tuple[str, ...]:
    mapping = json.loads(ESC50_LABELS_PATH.read_text(encoding="utf-8"))
    return tuple(mapping[str(index)] for index in range(50))


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
    full_model = len(model.classes_) == 50
    label_names = _esc50_label_names() if full_model else LABEL_NAMES
    return [
        {
            "label": DISPLAY_LABELS.get(label_names[int(model.classes_[index])], label_names[int(model.classes_[index])])
            if full_model else label_names[int(model.classes_[index])],
            "score": float(probabilities[index]),
        }
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

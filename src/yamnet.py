"""适合普通电脑运行的轻量级 YAMNet 音频识别封装。"""

from __future__ import annotations

import csv
import os
from math import gcd
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
YAMNET_URL = os.environ.get("YAMNET_URL", "https://tfhub.dev/google/yamnet/1")
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
CLASS_MAP_PATH = MODEL_DIR / "yamnet_class_map.csv"


@lru_cache(maxsize=1)
def load_yamnet():
    """加载并缓存预训练的 YAMNet 模型。"""
    return hub.load(YAMNET_URL)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """读取音频，转为单声道 float32，并统一到 16 kHz。"""
    waveform, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError("音频文件为空。")
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0:
        waveform = waveform / peak
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resample_poly(
            waveform, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE
    return waveform, sample_rate


def _download_class_map() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with urlopen(CLASS_MAP_URL, timeout=20) as response:
        CLASS_MAP_PATH.write_bytes(response.read())


@lru_cache(maxsize=1)
def class_names() -> tuple[str, ...]:
    """返回 YAMNet 的 521 个声音事件名称。"""
    if not CLASS_MAP_PATH.exists():
        try:
            _download_class_map()
        except Exception:
            return tuple(f"class_{index}" for index in range(521))
    try:
        with CLASS_MAP_PATH.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        names = tuple(row["display_name"] for row in rows)
        if len(names) == 521:
            return names
    except (OSError, KeyError, UnicodeError):
        pass
    return tuple(f"class_{index}" for index in range(521))


def predict_waveform(waveform: np.ndarray) -> dict:
    """运行 YAMNet，并汇总一段音频的逐帧识别结果。"""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("音频波形为空。")
    scores, embeddings, spectrogram = load_yamnet()(tf.convert_to_tensor(waveform))
    scores_np = np.asarray(scores)
    embeddings_np = np.asarray(embeddings)
    spectrogram_np = np.asarray(spectrogram)
    mean_scores = scores_np.mean(axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:5]
    names = class_names()
    top_results = [
        {
            "label": names[int(index)] if int(index) < len(names) else f"class_{index}",
            "score": float(mean_scores[index]),
            "index": int(index),
        }
        for index in top_indices
    ]
    return {
        "scores": scores_np,
        "embeddings": embeddings_np,
        "spectrogram": spectrogram_np,
        "top_results": top_results,
    }


def analyze_audio(path: str | Path) -> dict:
    """读取音频，并返回可视化数据和识别结果。"""
    waveform, sample_rate = load_audio(path)
    result = predict_waveform(waveform)
    result.update(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "duration": float(len(waveform) / sample_rate),
        }
    )
    return result

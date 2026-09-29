"""在命令行运行一次基础 YAMNet 音频识别。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from src.yamnet import TARGET_SAMPLE_RATE, analyze_audio


YAMNET_LABELS = {
    "Chainsaw": "电锯",
    "Helicopter": "直升机",
    "Rain": "雨声",
    "Sea waves": "海浪",
    "Crackling fire": "炉火噼啪声",
    "Crying, baby": "婴儿哭声",
    "Sneeze": "喷嚏",
    "Tick": "滴答声",
    "Rooster": "公鸡打鸣",
    "Dog": "狗叫声",
}


def create_demo_wav(path: Path) -> None:
    """生成一段短音调，便于立即检查完整识别流程。"""
    sample_count = TARGET_SAMPLE_RATE * 2
    timeline = np.arange(sample_count, dtype=np.float32) / TARGET_SAMPLE_RATE
    tone = 0.25 * np.sin(2 * np.pi * 440.0 * timeline)
    sf.write(str(path), tone, TARGET_SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 YAMNet 识别一个 WAV 或其他音频文件。")
    parser.add_argument("audio", nargs="?", type=Path, help="音频文件路径")
    args = parser.parse_args()

    audio_path = args.audio
    if audio_path is None:
        audio_path = Path("outputs") / "demo_tone.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if not audio_path.exists():
            create_demo_wav(audio_path)
        print(f"未提供输入文件，已生成演示音频：{audio_path}")
    if not audio_path.exists():
        raise SystemExit(f"找不到音频文件：{audio_path}")

    result = analyze_audio(audio_path)
    print(f"音频文件：{audio_path}")
    print(f"时长：{result['duration']:.2f} 秒，采样率：{result['sample_rate']} Hz")
    print(f"分析帧数：{result['scores'].shape[0]}")
    print(f"声音嵌入形状：{result['embeddings'].shape}")
    print("YAMNet 得分最高的 5 个声音事件：")
    for rank, item in enumerate(result["top_results"], start=1):
        label = YAMNET_LABELS.get(item["label"], item["label"])
        print(f"  {rank}. {label}：{item['score']:.4f}")


if __name__ == "__main__":
    main()

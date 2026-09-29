"""提取并缓存 ESC-10 或 ESC-50 的 MFCC 与 YAMNet 声音特征。"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import stft

from src.yamnet import analyze_audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "esc10_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "esc10_features_v2.npz"
CHECKPOINT_INTERVAL = 5


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    linear_step = 200.0 / 3
    log_step = np.log(6.4) / 27
    return np.where(
        frequencies < 1000,
        frequencies / linear_step,
        15 + np.log(np.maximum(frequencies, 1000) / 1000) / log_step,
    )


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    linear_step = 200.0 / 3
    log_step = np.log(6.4) / 27
    return np.where(
        mels < 15,
        mels * linear_step,
        1000 * np.exp((mels - 15) * log_step),
    )


@lru_cache(maxsize=4)
def _mel_filterbank(sample_rate: int) -> np.ndarray:
    frequencies = np.linspace(0, sample_rate / 2, 513)
    edges = _mel_to_hz(np.linspace(0, _hz_to_mel(np.array(sample_rate / 2)), 42))
    lower = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    upper = (edges[2:, None] - frequencies[None, :]) / (edges[2:, None] - edges[1:-1, None])
    filters = np.maximum(0, np.minimum(lower, upper))
    return filters * (2 / (edges[2:] - edges[:-2]))[:, None]


def mfcc_feature(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """返回 40 个 MFCC 系数的均值和标准差。"""
    _, _, spectrum = stft(
        waveform,
        fs=sample_rate,
        nperseg=1024,
        noverlap=512,
        boundary="even",
    )
    power = np.abs(spectrum * 512) ** 2
    mel_power = _mel_filterbank(sample_rate) @ power
    log_mel = 10 * np.log10(np.maximum(mel_power, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 80)
    values = dct(log_mel, type=2, axis=0, norm="ortho")
    return np.concatenate([values.mean(axis=1), values.std(axis=1)]).astype(np.float32)


def yamnet_embedding_feature(embeddings: np.ndarray) -> np.ndarray:
    """Pool frame embeddings with level, variation, and peak information."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1024:
        raise ValueError(f"YAMNet embeddings must have shape (frames, 1024), got {values.shape}")
    pooled = np.concatenate(
        [
            values.mean(axis=0),
            values.std(axis=0),
            values.max(axis=0),
        ]
    )
    return pooled.astype(np.float32)


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """以原子方式写入断点，避免中断造成缓存损坏。"""
    temporary = path.with_name(path.name + ".tmp")
    np.savez_compressed(temporary, **arrays)
    generated = Path(str(temporary) + ".npz")
    os.replace(generated, path)


def _checkpoint_arrays(
    mfcc_rows: list[np.ndarray],
    yamnet_rows: list[np.ndarray],
    kept_rows: list[dict],
    errors: list[dict],
) -> dict[str, np.ndarray]:
    return {
        "mfcc": np.stack(mfcc_rows) if mfcc_rows else np.empty((0, 80), dtype=np.float32),
        "yamnet": np.stack(yamnet_rows) if yamnet_rows else np.empty((0, 3072), dtype=np.float32),
        "records": np.asarray(
            [json.dumps(row, ensure_ascii=False) for row in kept_rows], dtype=np.str_
        ),
        "errors": np.asarray(
            [json.dumps(row, ensure_ascii=False) for row in errors], dtype=np.str_
        ),
    }


def _load_checkpoint(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], list[dict]]:
    if not path.exists():
        return [], [], [], []
    data = np.load(path, allow_pickle=False)
    mfcc_rows = [row.astype(np.float32) for row in data["mfcc"]]
    yamnet_rows = [row.astype(np.float32) for row in data["yamnet"]]
    kept_rows = [json.loads(value) for value in data["records"].astype(str)]
    errors = [json.loads(value) for value in data["errors"].astype(str)]
    return mfcc_rows, yamnet_rows, kept_rows, errors


def extract_features(manifest_path: Path, output_path: Path, limit: int | None = None) -> None:
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit).copy()
    checkpoint_path = output_path.with_name(output_path.stem + ".partial.npz")
    yamnet_rows, mfcc_rows, kept_rows, errors = ([], [], [], [])
    mfcc_rows, yamnet_rows, kept_rows, errors = _load_checkpoint(checkpoint_path)
    completed_paths = {row["path"] for row in kept_rows}
    completed_paths.update(row["path"] for row in errors)
    if completed_paths:
        print(f"正在从断点继续：已有 {len(completed_paths)} 个文件处理完成")
    started = time.perf_counter()
    total = len(manifest)
    for index, row in enumerate(manifest.to_dict("records"), start=1):
        if row["path"] in completed_paths:
            continue
        try:
            result = analyze_audio(row["path"])
            yamnet_rows.append(yamnet_embedding_feature(result["embeddings"]))
            mfcc_rows.append(mfcc_feature(result["waveform"], result["sample_rate"]))
            kept_rows.append(row)
        except Exception as error:  # keep an auditable record and continue
            errors.append({"path": row["path"], "error": repr(error)})
        completed_paths.add(row["path"])
        if len(completed_paths) % CHECKPOINT_INTERVAL == 0:
            _save_npz_atomic(checkpoint_path, **_checkpoint_arrays(mfcc_rows, yamnet_rows, kept_rows, errors))
        if index == 1 or index % 10 == 0 or index == total:
            elapsed = time.perf_counter() - started
            print(
                f"\r正在提取特征：{len(completed_paths)}/{total}（用时 {elapsed:.1f} 秒）",
                end="",
                flush=True,
            )
    print()
    if not kept_rows:
        raise RuntimeError("没有音频文件成功生成可用特征。")
    kept = pd.DataFrame(kept_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_npz_atomic(
        output_path,
        mfcc=np.stack(mfcc_rows),
        yamnet=np.stack(yamnet_rows),
        labels=kept["label"].to_numpy(dtype=np.int64),
        categories=np.asarray(kept["category"].tolist(), dtype=np.str_),
        splits=np.asarray(kept["split"].tolist(), dtype=np.str_),
        paths=np.asarray(kept["path"].tolist(), dtype=np.str_),
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "output": str(output_path),
                "samples": len(kept),
                "mfcc_shape": list(np.stack(mfcc_rows).shape),
                "yamnet_shape": list(np.stack(yamnet_rows).shape),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"特征已保存：{output_path}")
    print(f"MFCC 特征形状：{np.stack(mfcc_rows).shape}")
    print(f"YAMNet 特征形状：{np.stack(yamnet_rows).shape}")
    if errors:
        print(f"跳过 {len(errors)} 个文件，详情见：{metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    extract_features(args.manifest, args.output, args.limit)


if __name__ == "__main__":
    main()

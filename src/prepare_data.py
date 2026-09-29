"""下载 ESC-50，并按照官方 fold 生成十类或五十类数据清单。"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ESC50_ZIP = DATA_DIR / "esc50-master.zip"
ESC50_FULL_ZIP = DATA_DIR / "esc50-master-full.zip"
ESC10_CATEGORIES = (
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


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "yamnet-course-project/1.0"})
    with urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r正在下载 ESC-50：{downloaded / total:.0%}", end="")
    print()


def _find_metadata() -> Path | None:
    candidates = [
        DATA_DIR / "ESC-50-master" / "meta" / "esc50.csv",
        DATA_DIR / "ESC-50-master" / "meta" / "esc50.csv",
        DATA_DIR / "meta" / "esc50.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_dataset(download: bool = True) -> Path:
    metadata_path = _find_metadata()
    if metadata_path is not None:
        return metadata_path
    if not download:
        raise FileNotFoundError(
            "没有找到 ESC-50 元数据，请先运行 prepare_data.py 下载数据集。"
        )
    archive_path = ESC50_FULL_ZIP if ESC50_FULL_ZIP.exists() else ESC50_ZIP
    if not archive_path.exists():
        print(f"正在下载数据集：{ESC50_URL}")
        _download(ESC50_URL, archive_path)
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(
            f"下载的压缩包不完整或无效：{archive_path}。"
            "请重新运行数据下载命令。"
        )
    print(f"正在解压数据集：{archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(DATA_DIR)
    metadata_path = _find_metadata()
    if metadata_path is None:
        raise FileNotFoundError("下载的 ESC-50 压缩包中没有 meta/esc50.csv。")
    return metadata_path


def build_manifest(
    metadata_path: Path, output_path: Path, all_classes: bool = False
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required = {"filename", "fold", "target", "category", "esc10"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"ESC-50 元数据缺少字段：{sorted(missing)}")
    if all_classes:
        selected = metadata.copy()
        label_names = metadata.sort_values("target")["category"].drop_duplicates().tolist()
    else:
        is_esc10 = metadata["esc10"].astype(str).str.lower().isin({"true", "1", "yes"})
        selected = metadata[is_esc10 & metadata["category"].isin(ESC10_CATEGORIES)].copy()
        label_names = list(ESC10_CATEGORIES)
    selected["split"] = selected["fold"].map(
        {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
    )
    selected["path"] = selected["filename"].map(
        lambda name: str((metadata_path.parent.parent / "audio" / name).resolve())
    )
    selected["exists"] = selected["path"].map(lambda path: Path(path).exists())
    missing_files = selected[~selected["exists"]]
    if not missing_files.empty:
        names = ", ".join(missing_files["filename"].head(5).tolist())
        raise FileNotFoundError(f"缺少 {len(missing_files)} 个音频文件，示例：{names}")
    selected = selected.drop(columns=["exists"])
    selected = selected.sort_values(["split", "category", "filename"]).reset_index(drop=True)
    selected["label"] = pd.Categorical(
        selected["category"], categories=label_names, ordered=True
    ).codes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, encoding="utf-8")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--all-classes", action="store_true", help="使用全部 50 类声音")
    parser.add_argument(
        "--output", type=Path, default=None
    )
    args = parser.parse_args()
    metadata_path = ensure_dataset(download=not args.no_download)
    output = args.output or OUTPUT_DIR / (
        "esc50_manifest.csv" if args.all_classes else "esc10_manifest.csv"
    )
    manifest = build_manifest(metadata_path, output, all_classes=args.all_classes)
    print(f"数据清单：{output}")
    print(f"样本数量：{len(manifest)}")
    print("各数据划分和类别数量：")
    print(manifest.groupby(["split", "category"]).size().to_string())


if __name__ == "__main__":
    main()

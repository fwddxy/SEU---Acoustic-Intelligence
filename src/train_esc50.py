"""训练并评估 ESC-50 全部 50 类声音分类模型。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs" / "esc50_features.npz"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "esc50"


def _svm(c: float):
    return make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", probability=True, random_state=42))


def _mlp():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=350,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=42,
        ),
    )


def train_all(features_path: Path = FEATURES) -> dict:
    with np.load(features_path, allow_pickle=False) as data:
        mfcc = data["mfcc"]
        pooled = data["yamnet"]
        labels = data["labels"].astype(int)
        categories = data["categories"].astype(str)
        paths = data["paths"].astype(str)
        splits = data["splits"].astype(str)
    if pooled.shape[1] != 3072 or len(labels) != 2000:
        raise ValueError("ESC-50 特征缓存应包含 2000 条 3072 维 YAMNet 特征。")
    train, val, test = (splits == part for part in ("train", "val", "test"))
    if tuple(mask.sum() for mask in (train, val, test)) != (1200, 400, 400):
        raise ValueError("ESC-50 数据划分应为训练 1200、验证 400、测试 400。")
    manifest = pd.read_csv(ROOT / "outputs" / "esc50_manifest.csv").set_index("path")
    source_ids = manifest.loc[paths, "src_file"].astype(str).to_numpy()
    val = val & ~np.isin(source_ids, source_ids[test])
    if set(source_ids[train | val]) & set(source_ids[test]):
        raise ValueError("训练或验证音频与测试音频存在同源文件。")

    names = pd.DataFrame({"label": labels, "category": categories}).drop_duplicates().sort_values("label")
    if len(names) != 50 or names["label"].tolist() != list(range(50)):
        raise ValueError("ESC-50 标签映射不完整。")
    label_names = names["category"].tolist()
    features = {
        "mean": pooled[:, :1024],
        "stats": pooled,
        "mfcc": mfcc,
    }
    # Exclude validation clips sharing a source with the test fold.
    candidates = []
    for kind in ("mean", "stats"):
        for c in (1.0, 10.0):
            estimator = _svm(c)
            estimator.fit(features[kind][train], labels[train])
            predicted = estimator.predict(features[kind][val])
            candidates.append(
                {"feature": kind, "C": c, "validation_accuracy": float(accuracy_score(labels[val], predicted))}
            )
            print(f"YAMNet+SVM {kind} C={c:g}：验证准确率 {candidates[-1]['validation_accuracy']:.2%}", flush=True)
    best = max(candidates, key=lambda row: row["validation_accuracy"])
    specifications = {
        "yamnet_svm": (features[best["feature"]], _svm(best["C"]), best["feature"]),
        "yamnet_mlp": (features["mean"], _mlp(), "mean"),
        "mfcc_svm": (features["mfcc"], _svm(10.0), "mfcc"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "esc50_labels.json").write_text(
        json.dumps({str(index): name for index, name in enumerate(label_names)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {}
    fit = train | val
    for name, (x, estimator, feature_name) in specifications.items():
        estimator.fit(x[fit], labels[fit])
        predicted = estimator.predict(x[test])
        probabilities = estimator.predict_proba(x[test])
        class_ids = estimator.classes_
        matrix = confusion_matrix(labels[test], predicted, labels=np.arange(50))
        pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
            OUTPUT_DIR / f"{name}_confusion.csv", encoding="utf-8-sig"
        )
        test_indices = np.flatnonzero(test)
        ranks = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
        rows = []
        for row_index, index in enumerate(test_indices):
            top = ranks[row_index]
            rows.append(
                {
                    "path": paths[index],
                    "true_label": label_names[labels[index]],
                    "predicted_label": label_names[int(class_ids[top[0]])],
                    "top1_score": float(probabilities[row_index, top[0]]),
                    "top2_label": label_names[int(class_ids[top[1]])],
                    "top2_score": float(probabilities[row_index, top[1]]),
                    "top3_label": label_names[int(class_ids[top[2]])],
                    "top3_score": float(probabilities[row_index, top[2]]),
                }
            )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
        joblib.dump(estimator, MODEL_DIR / f"esc50_{name}.joblib")
        summary[name] = {
            "feature": feature_name,
            "train_samples": int(fit.sum()),
            "test_samples": int(test.sum()),
            "accuracy": float(accuracy_score(labels[test], predicted)),
            "macro_f1": float(f1_score(labels[test], predicted, average="macro", zero_division=0)),
        }
        print(f"{name}：50 类测试准确率 {summary[name]['accuracy']:.2%}", flush=True)
    summary["selection"] = {"validation_candidates": candidates, "best_yamnet_svm": best}
    summary["selection"]["validation_samples"] = int(val.sum())
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    train_all()

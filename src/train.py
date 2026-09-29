"""训练并评价三组 ESC-10 基线模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "outputs" / "esc10_features.npz"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
THRESHOLD_PATH = MODEL_DIR / "esc10_rejection_thresholds.json"
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


def _model_specs() -> dict:
    return {
        "mfcc_svm": (
            "mfcc",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", SVC(C=10, kernel="rbf", probability=True, random_state=42)),
                ]
            ),
        ),
        "yamnet_svm": (
            "yamnet",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", SVC(C=10, kernel="rbf", probability=True, random_state=42)),
                ]
            ),
        ),
        "yamnet_mlp": (
            "yamnet",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(128,),
                            activation="relu",
                            max_iter=300,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=20,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    }


def _save_confusion(name: str, matrix: np.ndarray) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        FIGURE_DIR / f"{name}_confusion_matrix.csv", encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(LABEL_NAMES)),
        yticks=np.arange(len(LABEL_NAMES)),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{name} confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(figure)


def train_all(features_path: Path) -> None:
    data = np.load(features_path, allow_pickle=False)
    splits = data["splits"].astype(str)
    labels = data["labels"].astype(np.int64)
    paths = data["paths"].astype(str)
    categories = data["categories"].astype(str)
    train_mask = splits == "train"
    test_mask = splits == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("特征缓存必须同时包含训练集和测试集样本。")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "esc10_labels.json").write_text(
        json.dumps({str(index): name for index, name in enumerate(LABEL_NAMES)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    all_metrics = {}
    rejection_thresholds = {}
    for name, (feature_name, model) in _model_specs().items():
        x = data[feature_name]
        model.fit(x[train_mask], labels[train_mask])
        predicted = model.predict(x[test_mask])
        probabilities = model.predict_proba(x[test_mask])
        if (splits == "val").any():
            val_mask = splits == "val"
            val_probabilities = model.predict_proba(x[val_mask])
            val_predicted = model.predict(x[val_mask])
            val_confidence = val_probabilities.max(axis=1)
            thresholds = np.linspace(0.20, 0.85, 66)
            valid = []
            for threshold in thresholds:
                selected = val_confidence >= threshold
                if selected.any():
                    precision = float((val_predicted[selected] == labels[val_mask][selected]).mean())
                    coverage = float(selected.mean())
                    if precision >= 0.95:
                        valid.append((coverage, -threshold, threshold))
            if valid:
                rejection_thresholds[name] = float(max(valid)[2])
            else:
                rejection_thresholds[name] = 0.50
        else:
            rejection_thresholds[name] = 0.50
        matrix = confusion_matrix(labels[test_mask], predicted, labels=np.arange(len(LABEL_NAMES)))
        metrics = {
            "model": name,
            "feature": feature_name,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "accuracy": float(accuracy_score(labels[test_mask], predicted)),
            "macro_precision": float(precision_score(labels[test_mask], predicted, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(labels[test_mask], predicted, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(labels[test_mask], predicted, average="macro", zero_division=0)),
            "classification_report": classification_report(
                labels[test_mask], predicted, labels=np.arange(len(LABEL_NAMES)), target_names=LABEL_NAMES, output_dict=True, zero_division=0
            ),
        }
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        (METRICS_DIR / f"{name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_confusion(name, matrix)
        test_indices = np.flatnonzero(test_mask)
        prediction_rows = []
        for row_index, sample_index in enumerate(test_indices):
            order = np.argsort(probabilities[row_index])[::-1][:3]
            prediction_rows.append(
                {
                    "path": paths[sample_index],
                    "true_label": LABEL_NAMES[labels[sample_index]],
                    "predicted_label": LABEL_NAMES[predicted[row_index]],
                    "top1_score": float(probabilities[row_index, order[0]]),
                    "top2_label": LABEL_NAMES[order[1]],
                    "top2_score": float(probabilities[row_index, order[1]]),
                    "top3_label": LABEL_NAMES[order[2]],
                    "top3_score": float(probabilities[row_index, order[2]]),
                }
            )
        pd.DataFrame(prediction_rows).to_csv(PREDICTION_DIR / f"{name}.csv", index=False, encoding="utf-8")
        all_metrics[name] = {key: value for key, value in metrics.items() if key != "classification_report"}
        display_name = {
            "mfcc_svm": "MFCC + 支持向量机",
            "yamnet_svm": "YAMNet + 支持向量机",
            "yamnet_mlp": "YAMNet + 多层感知机",
        }[name]
        print(f"{display_name}：准确率={metrics['accuracy']:.4f}，宏平均 F1={metrics['macro_f1']:.4f}")
    (METRICS_DIR / "summary.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_metrics).T.to_csv(METRICS_DIR / "summary.csv", encoding="utf-8")
    THRESHOLD_PATH.write_text(
        json.dumps(rejection_thresholds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"开放集拒识阈值已保存：{THRESHOLD_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    args = parser.parse_args()
    train_all(args.features)


if __name__ == "__main__":
    main()

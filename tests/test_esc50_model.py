"""Checks for the full ESC-50 model artifacts and label mapping."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.predict import DISPLAY_LABELS, MODEL_DIR


ROOT = Path(__file__).resolve().parents[1]


class Esc50ModelTests(unittest.TestCase):
    def test_manifest_has_50_classes_and_official_splits(self) -> None:
        manifest = pd.read_csv(ROOT / "outputs" / "esc50_manifest.csv")
        self.assertEqual(len(manifest), 2000)
        self.assertEqual(manifest["category"].nunique(), 50)
        self.assertEqual(manifest.groupby("split").size().to_dict(), {
            "train": 1200, "val": 400, "test": 400,
        })
        self.assertEqual(manifest.groupby(["split", "category"]).size().min(), 8)

    def test_deployed_models_and_chinese_labels_cover_all_classes(self) -> None:
        labels = json.loads((MODEL_DIR / "esc50_labels.json").read_text(encoding="utf-8"))
        self.assertEqual(len(labels), 50)
        self.assertEqual(set(labels.values()), set(DISPLAY_LABELS))
        for name, dimension in (
            ("mfcc_svm", 80),
            ("yamnet_svm", 3072),
            ("yamnet_mlp", 1024),
        ):
            with self.subTest(name=name):
                model = joblib.load(MODEL_DIR / f"esc50_{name}.joblib")
                self.assertEqual(model.n_features_in_, dimension)
                np.testing.assert_array_equal(model.classes_, np.arange(50))

    def test_reported_test_set_is_independent(self) -> None:
        manifest = pd.read_csv(ROOT / "outputs" / "esc50_manifest.csv")
        test_sources = set(manifest.loc[manifest["split"] == "test", "src_file"].astype(str))
        training_sources = set(manifest.loc[
            (manifest["split"] == "train")
            | ((manifest["split"] == "val") & ~manifest["src_file"].astype(str).isin(test_sources)),
            "src_file",
        ].astype(str))
        self.assertFalse(test_sources & training_sources)


if __name__ == "__main__":
    unittest.main()

"""Focused checks for the audio preprocessing used by training and inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.extract_features import mfcc_feature, yamnet_embedding_feature
from src.yamnet import TARGET_SAMPLE_RATE, load_audio


class AudioFeatureTests(unittest.TestCase):
    def test_resamples_esc50_rate_to_yamnet_rate(self) -> None:
        source_rate = 44_100
        timeline = np.arange(source_rate, dtype=np.float32) / source_rate
        waveform = 0.25 * np.sin(2 * np.pi * 440 * timeline)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        try:
            sf.write(path, waveform, source_rate)
            resampled, sample_rate = load_audio(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(sample_rate, TARGET_SAMPLE_RATE)
        self.assertEqual(resampled.shape, (TARGET_SAMPLE_RATE,))
        self.assertTrue(np.isfinite(resampled).all())

    def test_mfcc_has_80_finite_values_for_silence_and_tone(self) -> None:
        silence = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
        timeline = np.arange(TARGET_SAMPLE_RATE, dtype=np.float32) / TARGET_SAMPLE_RATE
        tone = np.sin(2 * np.pi * 440 * timeline).astype(np.float32)
        for waveform in (silence, tone):
            with self.subTest(silent=not waveform.any()):
                features = mfcc_feature(waveform, TARGET_SAMPLE_RATE)
                self.assertEqual(features.shape, (80,))
                self.assertEqual(features.dtype, np.float32)
                self.assertTrue(np.isfinite(features).all())

    def test_yamnet_embedding_pooling_has_3072_values(self) -> None:
        embeddings = np.ones((10, 1024), dtype=np.float32)
        features = yamnet_embedding_feature(embeddings)
        self.assertEqual(features.shape, (3072,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()

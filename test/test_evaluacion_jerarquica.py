import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluacion_jerarquica import (
    build_prediction_rows,
    category_metrics_per_class,
    export_category_confusion_matrix,
    export_evaluation_artifacts,
    format_label_set,
    instrument_metrics_per_class,
    sample_instrument_scores,
)


class CategoryMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        df = category_metrics_per_class(["a", "b"], ["a", "b"], ["a", "b"])

        self.assertEqual(df["precision"].tolist(), [1.0, 1.0])
        self.assertEqual(df["recall"].tolist(), [1.0, 1.0])
        self.assertEqual(df["f1"].tolist(), [1.0, 1.0])
        self.assertEqual(df["support"].tolist(), [1, 1])

    def test_incorrect_predictions(self):
        df = category_metrics_per_class(["a", "b"], ["b", "a"], ["a", "b"])

        self.assertEqual(df["precision"].tolist(), [0.0, 0.0])
        self.assertEqual(df["recall"].tolist(), [0.0, 0.0])
        self.assertEqual(df["f1"].tolist(), [0.0, 0.0])

    def test_class_without_predictions(self):
        df = category_metrics_per_class(["a", "b"], ["a", "a"], ["a", "b"])
        row_b = df.loc[df["class_name"] == "b"].iloc[0]

        self.assertEqual(row_b["precision"], 0.0)
        self.assertEqual(row_b["recall"], 0.0)
        self.assertEqual(row_b["f1"], 0.0)
        self.assertEqual(row_b["support"], 1)

    def test_category_csv_and_confusion_matrix_are_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            metrics_path = out_dir / "category_metrics_per_class.csv"
            matrix_path = out_dir / "category_confusion_matrix.png"

            category_metrics_per_class(["a", "b"], ["a", "a"], ["a", "b"]).to_csv(metrics_path, index=False)
            export_category_confusion_matrix(["a", "b"], ["a", "a"], ["a", "b"], matrix_path)

            self.assertTrue(metrics_path.exists())
            self.assertGreater(matrix_path.stat().st_size, 0)


class InstrumentMetricsTests(unittest.TestCase):
    def test_exact_match(self):
        scores = sample_instrument_scores(["guitarra"], ["guitarra"], ["guitarra"])

        self.assertEqual(scores["correct_instruments"], "guitarra")
        self.assertEqual(scores["missed_instruments"], "")
        self.assertEqual(scores["extra_instruments"], "")
        self.assertEqual(scores["instrument_f1"], 1.0)
        self.assertEqual(scores["instrument_jaccard"], 1.0)

    def test_missing_label(self):
        scores = sample_instrument_scores(["guitarra", "voz"], ["guitarra"], ["guitarra", "voz"])

        self.assertEqual(scores["correct_instruments"], "guitarra")
        self.assertEqual(scores["missed_instruments"], "voz")
        self.assertEqual(scores["extra_instruments"], "")
        self.assertAlmostEqual(scores["instrument_f1"], 2 / 3)
        self.assertAlmostEqual(scores["instrument_jaccard"], 1 / 2)

    def test_extra_label(self):
        scores = sample_instrument_scores(["guitarra"], ["guitarra", "voz"], ["guitarra", "voz"])

        self.assertEqual(scores["correct_instruments"], "guitarra")
        self.assertEqual(scores["missed_instruments"], "")
        self.assertEqual(scores["extra_instruments"], "voz")
        self.assertAlmostEqual(scores["instrument_f1"], 2 / 3)
        self.assertAlmostEqual(scores["instrument_jaccard"], 1 / 2)

    def test_empty_reference_and_prediction(self):
        scores = sample_instrument_scores([], [], ["guitarra", "voz"])

        self.assertEqual(scores["correct_instruments"], "")
        self.assertEqual(scores["missed_instruments"], "")
        self.assertEqual(scores["extra_instruments"], "")
        self.assertEqual(scores["instrument_f1"], 1.0)
        self.assertEqual(scores["instrument_jaccard"], 1.0)

    def test_instrument_metrics_csv_is_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instrument_metrics_per_class.csv"
            y_true = np.array([[1, 0], [0, 1]])
            y_pred = np.array([[1, 1], [0, 0]])

            df = instrument_metrics_per_class(y_true, y_pred, ["guitarra", "voz"])
            df.to_csv(path, index=False)

            self.assertTrue(path.exists())
            self.assertEqual(df["support"].tolist(), [1, 1])
            self.assertEqual(df.columns.tolist(), ["instrument_name", "precision", "recall", "f1", "support"])


class ExportTests(unittest.TestCase):
    def test_prediction_and_error_files_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported = export_evaluation_artifacts(
                y_true_category=["cat_a", "cat_b"],
                y_pred_category=["cat_a", "cat_a"],
                y_true_instruments=[["voz", "guitarra"], []],
                y_pred_instruments=[["guitarra"], ["voz"]],
                y_true_instruments_bin=np.array([[1, 1], [0, 0]]),
                y_pred_instruments_bin=np.array([[1, 0], [0, 1]]),
                category_order=["cat_a", "cat_b"],
                instrument_order=["guitarra", "voz"],
                metadata=[
                    {"video_id": "v1", "video_path": "a.mp4", "model_name": "m", "prompt_id": 0},
                    {"video_id": "v2", "video_path": "b.mp4", "model_name": "m", "prompt_id": 0},
                ],
                output_dir=Path(tmp) / "nested" / "out",
            )

            for path in exported.values():
                self.assertTrue(path.exists(), path)

            predictions = pd.read_csv(exported["evaluation_predictions"])
            category_errors = pd.read_csv(exported["category_errors"])
            instrument_errors = pd.read_csv(exported["instrument_errors"])

            self.assertEqual(predictions.loc[0, "true_instruments"], "guitarra|voz")
            self.assertEqual(predictions.loc[0, "correct_instruments"], "guitarra")
            self.assertEqual(predictions.loc[0, "missed_instruments"], "voz")
            self.assertEqual(predictions.loc[1, "extra_instruments"], "voz")
            self.assertEqual(category_errors["video_id"].tolist(), ["v2"])
            self.assertEqual(instrument_errors["video_id"].tolist(), ["v2", "v1"])

    def test_build_prediction_rows_validates_metadata_length(self):
        with self.assertRaises(ValueError):
            build_prediction_rows(["a"], ["a"], [[]], [[]], [], [])

    def test_multilabel_format_is_deterministic(self):
        self.assertEqual(format_label_set(["voz", "guitarra"], ["guitarra", "voz"]), "guitarra|voz")


if __name__ == "__main__":
    unittest.main()

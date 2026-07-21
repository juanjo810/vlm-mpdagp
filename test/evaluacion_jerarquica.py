import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

# ================= CONFIG =================
pred_excel = "./resultados_limpios_qwen_finetunedfull.xlsx"
real_excel = "../../Pruebas_finales/MUESTREO_BASE_SERPINS.xlsx"
prompt_indices = [0, 1, 2]

dict_dir = Path(__file__).resolve().parent  # carpeta donde estan los JSON
variant_to_id_json = "variant_to_id.json"
id_to_families_json = "id_to_families.json"
category_map_json = "category_map.json"
default_output_dir = "./evaluation_results"
# ==========================================


# ================= I/O DICCIONARIOS =================

def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dictionaries(base_dir: str | Path = dict_dir) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Load instrument and category dictionaries from the evaluation directory."""
    base_path = Path(base_dir)
    return (
        load_json(base_path / variant_to_id_json),
        load_json(base_path / id_to_families_json),
        load_json(base_path / category_map_json),
    )


VARIANT_TO_ID, ID_TO_FAMILIES, CATEGORY_MAP = load_dictionaries()


# ================= NORMALIZACION =================

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = strip_accents(text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_variants(text: str) -> str:
    """
    Normalizacion compatible con variant_to_id:
    - lowercase + sin tildes
    - convierte conectores en separador ';'
    - elimina signos raros
    """
    text = normalize_text(text)
    for c in [" e ", " y ", " and ", "+", ",", ";"]:
        text = text.replace(c, ";")
    text = re.sub(r"[^a-z0-9; ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_raw_labels(text) -> list[str]:
    """
    Split conservador por , ; y conectores.
    Devuelve tokens normalizados (strings) sin mapear.
    """
    t = normalize_for_variants(text)
    if not t:
        return []
    return [p.strip() for p in t.split(";") if p.strip()]


# ================= CATEGORIAS (tipo) =================

def normalize_category(cat: str) -> str:
    cat_n = normalize_text(cat)
    return CATEGORY_MAP.get(cat_n, cat_n)


# ================= INSTRUMENTOS (mapa a IDs) =================

def map_tokens_to_ids(tokens: list[str]) -> list[str]:
    """
    tokens: lista de strings ya normalizadas (sin tildes etc).
    Retorna lista de IDs canonicos (filtra desconocidos).
    """
    ids = []
    for tok in tokens:
        id_ = VARIANT_TO_ID.get(tok)
        if id_ is not None:
            ids.append(id_)
    return dedupe_preserving_order(ids)


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """Return values without duplicates while preserving their first occurrence."""
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def sorted_labels(labels: Iterable[str], order: Sequence[str] | None = None) -> list[str]:
    """Sort labels deterministically, respecting an official order when provided."""
    label_set = set(labels)
    if order is None:
        return sorted(label_set)
    ordered = [label for label in order if label in label_set]
    extras = sorted(label_set.difference(order))
    return ordered + extras


def format_label_set(labels: Iterable[str], order: Sequence[str] | None = None) -> str:
    """Format a multilabel set as a stable pipe-separated string."""
    return "|".join(sorted_labels(labels, order))


# ================= SIMILITUD JERARQUICA (opcional) =================
# Si quieres, aqui puedes enchufar tu scoring jerarquico usando ID_TO_FAMILIES
# para crear metricas "soft". Este script se centra en metricas estandar.

def safe_binary_kappa(y_true_col, y_pred_col):
    y_true_col = np.asarray(y_true_col).astype(int)
    y_pred_col = np.asarray(y_pred_col).astype(int)

    if np.all(y_true_col == y_true_col[0]) and np.all(y_pred_col == y_pred_col[0]):
        return 1.0 if y_true_col[0] == y_pred_col[0] else 0.0

    k = cohen_kappa_score(y_true_col, y_pred_col, labels=[0, 1])
    if np.isnan(k):
        return 1.0 if np.array_equal(y_true_col, y_pred_col) else 0.0
    return float(k)


def per_label_kappa(y_true_bin, y_pred_bin):
    kappas = [safe_binary_kappa(y_true_bin[:, j], y_pred_bin[:, j])
              for j in range(y_true_bin.shape[1])]
    return float(np.mean(kappas)), kappas


# ================= METRICAS =================

def validate_equal_length(name_a: str, values_a: Sequence[Any], name_b: str, values_b: Sequence[Any]) -> None:
    """Validate that two sample-aligned sequences have the same length."""
    if len(values_a) != len(values_b):
        raise ValueError(f"{name_a} y {name_b} no tienen el mismo numero de muestras.")


def category_class_order(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    official_categories: Sequence[str] | None = None,
) -> list[str]:
    """Build a deterministic category order for multiclass metrics."""
    observed = list(y_true) + list(y_pred)
    if official_categories is not None:
        return sorted_labels(observed, official_categories)
    category_map_values = sorted(set(CATEGORY_MAP.values()))
    return sorted_labels(observed + category_map_values, category_map_values)


def category_metrics_per_class(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_order: Sequence[str],
) -> pd.DataFrame:
    """Compute precision, recall, F1 and support for each category."""
    validate_equal_length("y_true", y_true, "y_pred", y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(class_order),
        target_names=list(class_order),
        output_dict=True,
        zero_division=0,
    )
    rows = [
        {
            "class_name": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1": report[class_name]["f1-score"],
            "support": int(report[class_name]["support"]),
        }
        for class_name in class_order
    ]
    return pd.DataFrame(rows, columns=["class_name", "precision", "recall", "f1", "support"])


def instrument_metrics_per_class(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    instrument_order: Sequence[str],
) -> pd.DataFrame:
    """Compute precision, recall, F1 and support for each instrument."""
    validate_equal_length("y_true", y_true_bin, "y_pred", y_pred_bin)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "instrument_name": list(instrument_order),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        },
        columns=["instrument_name", "precision", "recall", "f1", "support"],
    )


def multilabel_metrics(y_true, y_pred, all_ids, make_report=False, report_output_dict=False):
    """
    y_true/y_pred: list-of-list de IDs canonicos
    all_ids: vocabulario cerrado de IDs
    make_report: si True, genera classification_report
    report_output_dict: si True, report como dict; si False, como str
    """
    mlb = MultiLabelBinarizer(classes=all_ids)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    metrics = {}
    metrics["subset_accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
    metrics["hamming_loss"] = hamming_loss(y_true_bin, y_pred_bin)
    metrics["jaccard_samples"] = jaccard_score(
        y_true_bin, y_pred_bin, average="samples", zero_division=0
    )

    for avg in ["micro", "macro", "weighted", "samples"]:
        metrics[f"precision_{avg}"] = precision_score(
            y_true_bin, y_pred_bin, average=avg, zero_division=0
        )
        metrics[f"recall_{avg}"] = recall_score(
            y_true_bin, y_pred_bin, average=avg, zero_division=0
        )
        metrics[f"f1_{avg}"] = f1_score(
            y_true_bin, y_pred_bin, average=avg, zero_division=0
        )

    kappa_macro, kappas = per_label_kappa(y_true_bin, y_pred_bin)
    metrics["kappa_macro_labels"] = kappa_macro
    metrics["kappa_per_label"] = dict(zip(list(mlb.classes_), kappas))

    metrics["n_labels"] = len(mlb.classes_)

    report = None
    if make_report:
        report = classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=list(mlb.classes_),
            zero_division=0,
            output_dict=report_output_dict
        )

    return metrics, mlb, y_true_bin, y_pred_bin, report


def sample_instrument_scores(
    true_labels: Iterable[str],
    pred_labels: Iterable[str],
    instrument_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Compute per-sample multilabel details.

    Convention: when both reference and prediction are empty, F1 and Jaccard are 1.0
    because the sample is an exact empty-set match.
    """
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    correct = true_set & pred_set
    missed = true_set - pred_set
    extra = pred_set - true_set

    if not true_set and not pred_set:
        instrument_f1 = 1.0
        instrument_jaccard = 1.0
    else:
        denominator = (2 * len(correct)) + len(missed) + len(extra)
        instrument_f1 = (2 * len(correct) / denominator) if denominator else 0.0
        union = true_set | pred_set
        instrument_jaccard = len(correct) / len(union) if union else 1.0

    return {
        "correct_instruments": format_label_set(correct, instrument_order),
        "missed_instruments": format_label_set(missed, instrument_order),
        "extra_instruments": format_label_set(extra, instrument_order),
        "instrument_f1": instrument_f1,
        "instrument_jaccard": instrument_jaccard,
    }


def build_prediction_rows(
    y_true_category: Sequence[str],
    y_pred_category: Sequence[str],
    y_true_instruments: Sequence[Sequence[str]],
    y_pred_instruments: Sequence[Sequence[str]],
    metadata: Sequence[dict[str, Any]],
    instrument_order: Sequence[str],
) -> pd.DataFrame:
    """Build one row per evaluated sample with category and instrument details."""
    validate_equal_length("categorias reales", y_true_category, "categorias predichas", y_pred_category)
    validate_equal_length("instrumentos reales", y_true_instruments, "instrumentos predichos", y_pred_instruments)
    validate_equal_length("metadatos", metadata, "predicciones", y_pred_category)

    rows = []
    optional_columns = ["model_name", "model_variant", "prompt_id"]
    for idx, (true_cat, pred_cat, true_inst, pred_inst, sample_meta) in enumerate(
        zip(y_true_category, y_pred_category, y_true_instruments, y_pred_instruments, metadata)
    ):
        scores = sample_instrument_scores(true_inst, pred_inst, instrument_order)
        row = {
            "video_id": sample_meta.get("video_id", idx),
            "video_path": sample_meta.get("video_path", ""),
            "true_category": true_cat,
            "predicted_category": pred_cat,
            "category_correct": true_cat == pred_cat,
            "true_instruments": format_label_set(true_inst, instrument_order),
            "predicted_instruments": format_label_set(pred_inst, instrument_order),
            **scores,
        }
        for column in optional_columns:
            if column in sample_meta and sample_meta[column] is not None:
                row[column] = sample_meta[column]
        rows.append(row)

    base_columns = [
        "video_id",
        "video_path",
        "true_category",
        "predicted_category",
        "category_correct",
        "true_instruments",
        "predicted_instruments",
        "correct_instruments",
        "missed_instruments",
        "extra_instruments",
        "instrument_f1",
        "instrument_jaccard",
    ]
    extra_columns = [column for column in optional_columns if any(column in row for row in rows)]
    return pd.DataFrame(rows, columns=base_columns + extra_columns)


# ================= EXPORTACION =================

def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return an output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_category_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_order: Sequence[str],
    output_path: str | Path,
) -> None:
    """Save a row-normalized category confusion matrix as a PNG image."""
    validate_equal_length("y_true", y_true, "y_pred", y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(class_order), normalize="true")
    matrix = np.nan_to_num(matrix)

    label_count = max(1, len(class_order))
    fig_width = max(8.0, min(24.0, 0.55 * label_count + 4.0))
    fig_height = max(6.0, min(24.0, 0.55 * label_count + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(label_count),
        yticks=np.arange(label_count),
        xticklabels=class_order,
        yticklabels=class_order,
        ylabel="Categoria real",
        xlabel="Categoria predicha",
        title="Matriz de confusion normalizada por categoria real",
    )
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.tick_params(labelsize=8 if label_count > 12 else 10)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row_idx in range(label_count):
        for col_idx in range(label_count):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7 if label_count > 12 else 9,
            )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def export_evaluation_artifacts(
    y_true_category: Sequence[str],
    y_pred_category: Sequence[str],
    y_true_instruments: Sequence[Sequence[str]],
    y_pred_instruments: Sequence[Sequence[str]],
    y_true_instruments_bin: np.ndarray,
    y_pred_instruments_bin: np.ndarray,
    category_order: Sequence[str],
    instrument_order: Sequence[str],
    metadata: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export all requested evaluation CSV and PNG artifacts."""
    out_dir = ensure_output_dir(output_dir)

    category_metrics_path = out_dir / "category_metrics_per_class.csv"
    instrument_metrics_path = out_dir / "instrument_metrics_per_class.csv"
    confusion_matrix_path = out_dir / "category_confusion_matrix.png"
    predictions_path = out_dir / "evaluation_predictions.csv"
    category_errors_path = out_dir / "category_errors.csv"
    instrument_errors_path = out_dir / "instrument_errors.csv"

    category_metrics_per_class(y_true_category, y_pred_category, category_order).to_csv(
        category_metrics_path, index=False
    )
    instrument_metrics_per_class(
        y_true_instruments_bin, y_pred_instruments_bin, instrument_order
    ).to_csv(instrument_metrics_path, index=False)
    export_category_confusion_matrix(y_true_category, y_pred_category, category_order, confusion_matrix_path)

    predictions_df = build_prediction_rows(
        y_true_category,
        y_pred_category,
        y_true_instruments,
        y_pred_instruments,
        metadata,
        instrument_order,
    )
    predictions_df.to_csv(predictions_path, index=False)

    category_error_columns = [
        "video_id",
        "video_path",
        "true_category",
        "predicted_category",
        *[column for column in ["model_name", "model_variant", "prompt_id"] if column in predictions_df.columns],
    ]
    predictions_df.loc[
        predictions_df["true_category"] != predictions_df["predicted_category"],
        category_error_columns,
    ].to_csv(category_errors_path, index=False)

    instrument_error_columns = [
        "video_id",
        "video_path",
        "true_instruments",
        "predicted_instruments",
        "correct_instruments",
        "missed_instruments",
        "extra_instruments",
        "instrument_f1",
        "instrument_jaccard",
        *[column for column in ["model_name", "model_variant", "prompt_id"] if column in predictions_df.columns],
    ]
    instrument_errors = predictions_df.loc[
        predictions_df["true_instruments"] != predictions_df["predicted_instruments"],
        instrument_error_columns,
    ].sort_values(["instrument_f1", "instrument_jaccard"], ascending=[True, True])
    instrument_errors.to_csv(instrument_errors_path, index=False)

    return {
        "category_metrics_per_class": category_metrics_path,
        "instrument_metrics_per_class": instrument_metrics_path,
        "category_confusion_matrix": confusion_matrix_path,
        "evaluation_predictions": predictions_path,
        "category_errors": category_errors_path,
        "instrument_errors": instrument_errors_path,
    }


# ================= CARGA Y EVALUACION =================

def metadata_from_dataframes(
    df_pred: pd.DataFrame,
    df_real: pd.DataFrame,
    prompt_id: int,
) -> list[dict[str, Any]]:
    """Build sample metadata from available prediction/ground-truth columns."""
    validate_equal_length("predicciones", df_pred, "etiquetas reales", df_real)
    metadata_rows = []
    id_candidates = ["video_id", "id", "ID", "Video_ID", "video"]
    path_candidates = ["video_path", "path", "Video", "Link", "link", "URL", "url"]
    model_candidates = ["model_name", "modelo", "MODEL_NAME"]
    variant_candidates = ["model_variant", "variant", "variante"]

    def first_available(row_pred: pd.Series, row_real: pd.Series, candidates: Sequence[str], default: Any) -> Any:
        for column in candidates:
            if column in row_pred and not pd.isna(row_pred[column]):
                return row_pred[column]
            if column in row_real and not pd.isna(row_real[column]):
                return row_real[column]
        return default

    for idx in range(len(df_pred)):
        pred_row = df_pred.iloc[idx]
        real_row = df_real.iloc[idx]
        meta = {
            "video_id": first_available(pred_row, real_row, id_candidates, idx),
            "video_path": first_available(pred_row, real_row, path_candidates, ""),
            "prompt_id": prompt_id,
        }
        model_name = first_available(pred_row, real_row, model_candidates, None)
        model_variant = first_available(pred_row, real_row, variant_candidates, None)
        if model_name is not None:
            meta["model_name"] = model_name
        if model_variant is not None:
            meta["model_variant"] = model_variant
        metadata_rows.append(meta)
    return metadata_rows


def prepare_evaluation_dataframe(df_pred: pd.DataFrame, df_real: pd.DataFrame) -> pd.DataFrame:
    """Combine predictions and ground truth with normalized reference labels."""
    validate_equal_length("predicciones", df_pred, "etiquetas reales", df_real)
    df = pd.concat([df_pred.reset_index(drop=True), df_real.reset_index(drop=True)], axis=1)
    df["Categorias_real"] = df["Categorias"].apply(normalize_category)
    df["Instrumentos_real_ids"] = df["Instrumentos"].apply(lambda x: map_tokens_to_ids(split_raw_labels(x)))
    return df


def evaluate_prompt(
    df: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_real: pd.DataFrame,
    prompt_id: int,
    output_dir: str | Path,
    all_instrument_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one prompt and export requested artifacts without changing global metrics."""
    tipo_col = f"tipo_{prompt_id}"
    inst_col = f"instrumentos_{prompt_id}"
    if tipo_col not in df.columns:
        raise KeyError(f"No existe {tipo_col}")
    if inst_col not in df.columns:
        raise KeyError(f"No existe {inst_col}")

    category_pred_col = f"tipo_pred_{prompt_id}"
    instrument_pred_col = f"inst_pred_ids_{prompt_id}"
    df[category_pred_col] = df[tipo_col].apply(normalize_category)
    df[instrument_pred_col] = df[inst_col].apply(lambda x: map_tokens_to_ids(split_raw_labels(x)))

    y_true_category = df["Categorias_real"].tolist()
    y_pred_category = df[category_pred_col].tolist()
    y_true_instruments = df["Instrumentos_real_ids"].tolist()
    y_pred_instruments = df[instrument_pred_col].tolist()

    instrument_order = list(all_instrument_ids or sorted(ID_TO_FAMILIES.keys()))
    metrics, mlb, y_true_bin, y_pred_bin, report = multilabel_metrics(
        y_true_instruments, y_pred_instruments, all_ids=instrument_order, make_report=False
    )
    category_order = category_class_order(y_true_category, y_pred_category)
    metadata = metadata_from_dataframes(df_pred, df_real, prompt_id)
    exported_paths = export_evaluation_artifacts(
        y_true_category=y_true_category,
        y_pred_category=y_pred_category,
        y_true_instruments=y_true_instruments,
        y_pred_instruments=y_pred_instruments,
        y_true_instruments_bin=y_true_bin,
        y_pred_instruments_bin=y_pred_bin,
        category_order=category_order,
        instrument_order=list(mlb.classes_),
        metadata=metadata,
        output_dir=output_dir,
    )

    return {
        "tipo_acc": (df["Categorias_real"] == df[category_pred_col]).mean(),
        "instrument_metrics": metrics,
        "mlb": mlb,
        "y_true_bin": y_true_bin,
        "y_pred_bin": y_pred_bin,
        "report": report,
        "exported_paths": exported_paths,
    }


def print_global_metrics(df: pd.DataFrame, prompt_id: int, evaluation: dict[str, Any]) -> None:
    """Print the same global metric meanings as the original script."""
    print(f"\n=========== PROMPT {prompt_id} ===========")
    print(f"Tipo accuracy: {evaluation['tipo_acc']*100:.2f}%")

    pred_col = f"tipo_pred_{prompt_id}"
    for avg in ["micro", "macro", "weighted"]:
        f1 = f1_score(df["Categorias_real"], df[pred_col], average=avg, zero_division=0)
        prec = precision_score(df["Categorias_real"], df[pred_col], average=avg, zero_division=0)
        rec = recall_score(df["Categorias_real"], df[pred_col], average=avg, zero_division=0)
        print(f"Tipo precision ({avg}): {prec*100:.2f}%")
        print(f"Tipo recall    ({avg}): {rec*100:.2f}%")
        print(f"Tipo F1        ({avg}): {f1*100:.2f}%")

    kappa_tipo = cohen_kappa_score(df["Categorias_real"], df[pred_col])
    print(f"Tipo Cohen kappa: {kappa_tipo:.4f}")

    m = evaluation["instrument_metrics"]
    print(f"Instrumentos subset_accuracy (exact match): {m['subset_accuracy']*100:.2f}%")
    print(f"Instrumentos hamming_loss (lower=better): {m['hamming_loss']:.4f}")
    print(f"Instrumentos jaccard_samples: {m['jaccard_samples']*100:.2f}%")
    print(f"Instrumentos precision_micro: {m['precision_micro']*100:.2f}%")
    print(f"Instrumentos recall_micro:    {m['recall_micro']*100:.2f}%")
    print(f"Instrumentos f1_micro:        {m['f1_micro']*100:.2f}%")
    print(f"Instrumentos precision_macro: {m['precision_macro']*100:.2f}%")
    print(f"Instrumentos recall_macro:    {m['recall_macro']*100:.2f}%")
    print(f"Instrumentos f1_macro:        {m['f1_macro']*100:.2f}%")
    print(f"Instrumentos f1_weighted:     {m['f1_weighted']*100:.2f}%")
    print(f"Instrumentos f1_samples:      {m['f1_samples']*100:.2f}%")
    print(f"Instrumentos kappa_macro_labels: {m['kappa_macro_labels']:.4f}")
    print("Ficheros exportados:")
    for path in evaluation["exported_paths"].values():
        print(f"  - {path}")


def parse_prompt_indices(value: str) -> list[int]:
    """Parse comma-separated prompt indices."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua categorias e instrumentos y exporta resultados.")
    parser.add_argument("--pred-excel", default=pred_excel, help="Excel con predicciones limpias.")
    parser.add_argument("--real-excel", default=real_excel, help="Excel con etiquetas reales.")
    parser.add_argument(
        "--prompt-indices",
        default=",".join(str(i) for i in prompt_indices),
        help="Indices de prompt separados por coma, por ejemplo: 0,1,2.",
    )
    parser.add_argument("--dict-dir", default=dict_dir, help="Directorio con los JSON de normalizacion.")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directorio base de resultados.")
    args = parser.parse_args()

    global VARIANT_TO_ID, ID_TO_FAMILIES, CATEGORY_MAP
    VARIANT_TO_ID, ID_TO_FAMILIES, CATEGORY_MAP = load_dictionaries(args.dict_dir)

    df_pred = pd.read_excel(args.pred_excel)
    df_real = pd.read_excel(args.real_excel)
    df = prepare_evaluation_dataframe(df_pred, df_real)

    selected_prompts = parse_prompt_indices(args.prompt_indices)
    for prompt_id in selected_prompts:
        tipo_col = f"tipo_{prompt_id}"
        inst_col = f"instrumentos_{prompt_id}"
        if tipo_col not in df.columns:
            print(f"No existe {tipo_col}")
            continue
        if inst_col not in df.columns:
            print(f"No existe {inst_col}")
            continue

        prompt_output_dir = Path(args.output_dir) / f"prompt_{prompt_id}"
        evaluation = evaluate_prompt(
            df=df,
            df_pred=df_pred,
            df_real=df_real,
            prompt_id=prompt_id,
            output_dir=prompt_output_dir,
            all_instrument_ids=sorted(ID_TO_FAMILIES.keys()),
        )
        print_global_metrics(df, prompt_id, evaluation)


if __name__ == "__main__":
    main()

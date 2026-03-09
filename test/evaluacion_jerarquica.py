import pandas as pd
import re
import unicodedata
import json
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,          # subset accuracy en multilabel
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    jaccard_score,
    cohen_kappa_score,
    classification_report
)

import numpy as np

# ================= CONFIG =================
pred_excel = "./resultados_limpios_qwen_finetunedfull.xlsx"
real_excel = "../../Pruebas_finales/MUESTREO_BASE_SERPINS.xlsx"
prompt_indices = [0, 1, 2]

dict_dir = "."  # carpeta donde están los JSON
variant_to_id_json = "variant_to_id.json"
id_to_families_json = "id_to_families.json"
category_map_json = "category_map.json"
# ==========================================


# ================= I/O DICCIONARIOS =================

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

VARIANT_TO_ID = load_json(str(Path(dict_dir) / variant_to_id_json))
ID_TO_FAMILIES = load_json(str(Path(dict_dir) / id_to_families_json))
CATEGORY_MAP = load_json(str(Path(dict_dir) / category_map_json))


# ================= NORMALIZACIÓN =================

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
    # colapsar espacios
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_for_variants(text: str) -> str:
    """
    Normalización compatible con variant_to_id:
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

def split_raw_labels(text) -> list:
    """
    Split conservador por , ; y conectores.
    Devuelve tokens normalizados (strings) sin mapear.
    """
    t = normalize_for_variants(text)
    if not t:
        return []
    parts = [p.strip() for p in t.split(";") if p.strip()]
    return parts


# ================= CATEGORÍAS (tipo) =================

def normalize_category(cat: str) -> str:
    cat_n = normalize_text(cat)
    return CATEGORY_MAP.get(cat_n, cat_n)


# ================= INSTRUMENTOS (mapa a IDs) =================

def map_tokens_to_ids(tokens: list) -> list:
    """
    tokens: lista de strings ya normalizadas (sin tildes etc).
    Retorna lista de IDs canónicos (filtra desconocidos).
    """
    ids = []
    for tok in tokens:
        id_ = VARIANT_TO_ID.get(tok)
        if id_ is not None:
            ids.append(id_)
        # si no está, lo ignoramos (o podrías meter 'unknown')
    # dedup estable
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ================= SIMILITUD JERÁRQUICA (opcional) =================
# Si quieres, aquí puedes enchufar tu scoring jerárquico usando ID_TO_FAMILIES
# para crear métricas "soft". Este script se centra en métricas estándar.

def safe_binary_kappa(y_true_col, y_pred_col):
    y_true_col = np.asarray(y_true_col).astype(int)
    y_pred_col = np.asarray(y_pred_col).astype(int)

    # Caso degenerado: ambos constantes y iguales
    if np.all(y_true_col == y_true_col[0]) and np.all(y_pred_col == y_pred_col[0]):
        return 1.0 if y_true_col[0] == y_pred_col[0] else 0.0

    k = cohen_kappa_score(y_true_col, y_pred_col, labels=[0, 1])  # fija forma [web:55]
    if np.isnan(k):
        # fallback conservador
        return 1.0 if np.array_equal(y_true_col, y_pred_col) else 0.0
    return float(k)

def per_label_kappa(y_true_bin, y_pred_bin):
    kappas = [safe_binary_kappa(y_true_bin[:, j], y_pred_bin[:, j])
              for j in range(y_true_bin.shape[1])]
    return float(np.mean(kappas)), kappas



# ================= CARGA DE DATOS =================

df_pred = pd.read_excel(pred_excel)
df_real = pd.read_excel(real_excel)

if len(df_pred) != len(df_real):
    raise ValueError("Los Excel no tienen el mismo número de filas.")

df = pd.concat([df_pred, df_real], axis=1)

# Reales
df["Categorias_real"] = df["Categorias"].apply(normalize_category)

# Instrumentos reales → IDs
df["Instrumentos_real_ids"] = df["Instrumentos"].apply(lambda x: map_tokens_to_ids(split_raw_labels(x)))


# ================= EVALUACIÓN =================

def multilabel_metrics(y_true, y_pred, all_ids, make_report=False, report_output_dict=False):
    """
    y_true/y_pred: list-of-list de IDs canónicos
    all_ids: vocabulario cerrado de IDs (evita 'unknown class(es) will be ignored') [web:66]
    make_report: si True, genera classification_report
    report_output_dict: si True, report como dict; si False, como str [web:71]
    """
    mlb = MultiLabelBinarizer(classes=all_ids)  # fija espacio de etiquetas [web:66]
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    metrics = {}
    metrics["subset_accuracy"] = accuracy_score(y_true_bin, y_pred_bin)  # [web:42]
    metrics["hamming_loss"] = hamming_loss(y_true_bin, y_pred_bin)       # [web:41]
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

    # Kappa multilabel (macro sobre etiquetas binarias) [web:43]
    kappa_macro, kappas = per_label_kappa(y_true_bin, y_pred_bin)  # [web:55]
    metrics["kappa_macro_labels"] = kappa_macro
    metrics["kappa_per_label"] = dict(zip(list(mlb.classes_), kappas))  # útil para debug

    metrics["n_labels"] = len(mlb.classes_)

    report = None
    if make_report:
        report = classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=list(mlb.classes_),
            zero_division=0,
            output_dict=report_output_dict
        )  # [web:71]

    return metrics, mlb, y_true_bin, y_pred_bin, report


for i in prompt_indices:
    print(f"\n=========== PROMPT {i} ===========")

    tipo_col = f"tipo_{i}"
    inst_col = f"instrumentos_{i}"

    if tipo_col not in df.columns:
        print(f"No existe {tipo_col}")
        continue
    if inst_col not in df.columns:
        print(f"No existe {inst_col}")
        continue

    # ---- TIPO (single-label) ----
    df[f"tipo_pred_{i}"] = df[tipo_col].apply(normalize_category)

    tipo_acc = (df["Categorias_real"] == df[f"tipo_pred_{i}"]).mean()
    print(f"Tipo accuracy: {tipo_acc*100:.2f}%")

    # F1/Prec/Rec para single-label multiclass
    # (en sklearn se calculan igual; usa average micro/macro/weighted)
    # micro en multiclass equivale a accuracy si no hay multilabel
    for avg in ["micro", "macro", "weighted"]:
        f1 = f1_score(df["Categorias_real"], df[f"tipo_pred_{i}"], average=avg, zero_division=0)
        prec = precision_score(df["Categorias_real"], df[f"tipo_pred_{i}"], average=avg, zero_division=0)
        rec = recall_score(df["Categorias_real"], df[f"tipo_pred_{i}"], average=avg, zero_division=0)
        print(f"Tipo precision ({avg}): {prec*100:.2f}%")
        print(f"Tipo recall    ({avg}): {rec*100:.2f}%")
        print(f"Tipo F1        ({avg}): {f1*100:.2f}%")

    # Kappa para single-label sí aplica directamente [web:55]
    kappa_tipo = cohen_kappa_score(df["Categorias_real"], df[f"tipo_pred_{i}"])
    print(f"Tipo Cohen kappa: {kappa_tipo:.4f}")

    # ---- INSTRUMENTOS (multi-label) ----
    # pred → tokens → IDs
    df[f"inst_pred_ids_{i}"] = df[inst_col].apply(lambda x: map_tokens_to_ids(split_raw_labels(x)))

    y_true = df["Instrumentos_real_ids"].tolist()
    y_pred = df[f"inst_pred_ids_{i}"].tolist()

    ALL_IDS = sorted(ID_TO_FAMILIES.keys())  # vocabulario cerrado recomendado
    m, mlb, y_true_bin, y_pred_bin, report = multilabel_metrics(
        y_true, y_pred, all_ids=ALL_IDS, make_report=False
    )


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

    # Report detallado por etiqueta (opcional; puede ser largo)
    # print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0))

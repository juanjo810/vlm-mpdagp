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
    for c in [" e ", " y ", " and ", "+", ",", ";", "/"]:
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


# ================= SIMILITUD JERÁRQUICA =================

def families_for_ids(inst_ids: list[str]) -> list[str]:
    families = []
    seen = set()
    for inst_id in inst_ids:
        for family in ID_TO_FAMILIES.get(inst_id, []):
            if family not in seen:
                families.append(family)
                seen.add(family)
    return families

def hierarchy_similarity_id(pred_id: str, true_id: str) -> float:
    """
    Similaridad jerárquica entre dos instrumentos (0..1):
    - 1.0 cuando el instrumento canónico coincide
    - en otro caso, Jaccard entre familias organológicas
    """
    if pred_id == true_id:
        return 1.0

    pred_families = set(ID_TO_FAMILIES.get(pred_id, []))
    true_families = set(ID_TO_FAMILIES.get(true_id, []))

    if not pred_families and not true_families:
        return 0.0

    union = pred_families | true_families
    if not union:
        return 0.0
    intersection = pred_families & true_families
    return len(intersection) / len(union)

def soft_prf_single_example(true_ids: list[str], pred_ids: list[str]) -> tuple[float, float, float]:
    """
    Scores jerárquicos por ejemplo:
    - precision_soft: media del mejor match de cada predicción contra verdad
    - recall_soft: media del mejor match de cada real contra predicción
    """
    if len(pred_ids) == 0 and len(true_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_ids) == 0 or len(true_ids) == 0:
        return 0.0, 0.0, 0.0

    pred_best = [max(hierarchy_similarity_id(pred, true) for true in true_ids) for pred in pred_ids]
    true_best = [max(hierarchy_similarity_id(pred, true) for pred in pred_ids) for true in true_ids]

    precision_soft = float(np.mean(pred_best)) if pred_best else 0.0
    recall_soft = float(np.mean(true_best)) if true_best else 0.0

    if precision_soft + recall_soft == 0:
        f1_soft = 0.0
    else:
        f1_soft = 2 * precision_soft * recall_soft / (precision_soft + recall_soft)

    return precision_soft, recall_soft, f1_soft

def hierarchical_instrument_metrics(y_true_ids: list[list[str]], y_pred_ids: list[list[str]]) -> dict:
    """
    Métricas de evaluación jerárquica:
    1) Soft precision/recall/F1 por similitud ID->familias
    2) Multilabel clásico, pero en el espacio de familias organológicas
    """
    per_example_scores = [soft_prf_single_example(true, pred) for true, pred in zip(y_true_ids, y_pred_ids)]
    precision_soft = float(np.mean([s[0] for s in per_example_scores]))
    recall_soft = float(np.mean([s[1] for s in per_example_scores]))
    f1_soft = float(np.mean([s[2] for s in per_example_scores]))

    y_true_families = [families_for_ids(ids) for ids in y_true_ids]
    y_pred_families = [families_for_ids(ids) for ids in y_pred_ids]

    all_families = sorted({family for families in ID_TO_FAMILIES.values() for family in families})
    fam_mlb = MultiLabelBinarizer(classes=all_families)
    y_true_fam_bin = fam_mlb.fit_transform(y_true_families)
    y_pred_fam_bin = fam_mlb.transform(y_pred_families)

    return {
        "soft_precision": precision_soft,
        "soft_recall": recall_soft,
        "soft_f1": f1_soft,
        "families_subset_accuracy": accuracy_score(y_true_fam_bin, y_pred_fam_bin),
        "families_hamming_loss": hamming_loss(y_true_fam_bin, y_pred_fam_bin),
        "families_jaccard_samples": jaccard_score(y_true_fam_bin, y_pred_fam_bin, average="samples", zero_division=0),
        "families_precision_micro": precision_score(y_true_fam_bin, y_pred_fam_bin, average="micro", zero_division=0),
        "families_recall_micro": recall_score(y_true_fam_bin, y_pred_fam_bin, average="micro", zero_division=0),
        "families_f1_micro": f1_score(y_true_fam_bin, y_pred_fam_bin, average="micro", zero_division=0),
    }

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
    h = hierarchical_instrument_metrics(y_true, y_pred)


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
    print(f"Instrumentos soft_precision jerárquica: {h['soft_precision']*100:.2f}%")
    print(f"Instrumentos soft_recall jerárquica:    {h['soft_recall']*100:.2f}%")
    print(f"Instrumentos soft_f1 jerárquica:        {h['soft_f1']*100:.2f}%")
    print(f"Familias subset_accuracy (exact match): {h['families_subset_accuracy']*100:.2f}%")
    print(f"Familias hamming_loss (lower=better):   {h['families_hamming_loss']:.4f}")
    print(f"Familias jaccard_samples:               {h['families_jaccard_samples']*100:.2f}%")
    print(f"Familias precision_micro:               {h['families_precision_micro']*100:.2f}%")
    print(f"Familias recall_micro:                  {h['families_recall_micro']*100:.2f}%")
    print(f"Familias f1_micro:                      {h['families_f1_micro']*100:.2f}%")

    # Report detallado por etiqueta (opcional; puede ser largo)
    # print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0))

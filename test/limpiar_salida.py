import pandas as pd
import re

# ========= CONFIG =========
input_excel = "resultado_qwen_finetunedfull.xlsx"
output_excel = "resultados_limpios_qwen_finetunedfull.xlsx"

# Columnas donde están los outputs de los prompts
prompt_columns = [
    "prompt_0",
    "prompt_1",
    "prompt_2"
]
# ===========================


def clean_assistant_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Extraer todo lo que viene después de "assistant"
    match = re.search(r'assistant\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_fields(text):
    """
    Extrae:
    Tipo
    Instrumentos
    Personas
    Ambiente
    """
    tipo = instrumentos = personas = ambiente = ""

    tipo_match = re.search(r"Tipo:\s*(.*)", text, re.IGNORECASE)
    inst_match = re.search(r"Instrumentos:\s*(.*)", text, re.IGNORECASE)
    pers_match = re.search(r"Personas:\s*(.*)", text, re.IGNORECASE)
    amb_match = re.search(r"Ambiente:\s*(.*)", text, re.IGNORECASE)

    if tipo_match:
        tipo = tipo_match.group(1).strip()
    if inst_match:
        instrumentos = inst_match.group(1).strip()
    if pers_match:
        personas = pers_match.group(1).strip()
    if amb_match:
        ambiente = amb_match.group(1).strip()

    return tipo, instrumentos, personas, ambiente


# Leer Excel
df = pd.read_excel(input_excel)

for i, col in enumerate(prompt_columns):

    if col not in df.columns:
        print(f"⚠ La columna {col} no existe. Saltando.")
        continue

    # Limpiar texto
    cleaned = df[col].apply(clean_assistant_text)

    # Extraer campos
    extracted = cleaned.apply(extract_fields)

    df[f"tipo_{i}"] = extracted.apply(lambda x: x[0])
    df[f"instrumentos_{i}"] = extracted.apply(lambda x: x[1])
    df[f"personas_{i}"] = extracted.apply(lambda x: x[2])
    df[f"ambiente_{i}"] = extracted.apply(lambda x: x[3])

# Guardar nuevo Excel
df.to_excel(output_excel, index=False)

print(f"Excel procesado guardado como: {output_excel}")
# vlm-mpdagp

Repository for **preprocessing, fine-tuning, inference, and evaluation** of a Vision-Language Model (VLM) focused on Portuguese traditional music videos, using **Qwen3-VL + Unsloth**.

---

## 1) Repository purpose

This project provides an end-to-end workflow to:

1. Build a dataset from an Excel file containing Vimeo links.
2. Extract video frames and generate conversational JSONL samples (`messages` format).
3. Fine-tune a Qwen3-VL model with LoRA.
4. Run batch inference over videos.
5. Clean model outputs and evaluate predictions (category + instruments).

---

## 2) Repository structure

```text
vlm-mpdagp/
├── README.md
├── requirements.txt
├── dataset/
│   └── Base_pruebas.xlsx
├── preprocessing/
│   └── convert_songs.py
├── train/
│   ├── train_qwen.py
│   └── Qwen3_VL_(8B)_Vision.ipynb
└── test/
    ├── test_model.py
    ├── test_unsloth.py
    ├── limpiar_salida.py
    ├── evaluacion_jerarquica.py
    ├── category_map.json
    ├── variant_to_id.json
    └── id_to_families.json
```

---

## 3) Setup and requirements

### 3.1 Python dependencies

Install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> Note: this project targets CUDA GPU environments and includes VLM training/inference libraries (`torch`, `transformers`, `unsloth`, `bitsandbytes`, etc.).

### 3.2 System dependencies

- **ffmpeg** (used in `test/test_model.py` to trim videos).
- Internet access (Vimeo downloads + Hugging Face checkpoints).

### 3.3 Sensitive configuration

Some scripts currently contain hardcoded tokens/paths. Recommended environment variables:

- `VIMEO_ACCESS_TOKEN`
- input/output dataset paths
- local/remote HF model names

---

## 4) Directory and file documentation

## `/`

### `README.md`
Main project documentation.

### `requirements.txt`
Complete dependency list for training, inference, and evaluation, including:

- Deep learning: `torch`, `transformers`, `trl`, `peft`, `unsloth`
- Data and metrics: `pandas`, `scikit-learn`, `datasets`
- Video and image processing: `moviepy`, `Pillow`, `imageio-ffmpeg`

---

## `/dataset`

### `Base_pruebas.xlsx`
Base dataset in Excel format used by preprocessing.

According to `preprocessing/convert_songs.py`, required columns are:

- `Link` (Vimeo URL)
- `Categorias`
- `Instrumentos`

---

## `/preprocessing`

### `convert_songs.py`
Main dataset builder for fine-tuning.

#### What it does

1. Reads the input Excel file.
2. For each Vimeo video:
   - gets the download URL from Vimeo API,
   - downloads the video,
   - selects 2 heuristic clip windows,
   - extracts 8 frames per clip (16 total),
   - resizes all frames to 448x448,
   - builds one multimodal training sample in `messages` format.
3. Splits data into train/test.
4. Writes:
   - `preprocessed_dataset/train.jsonl`
   - `preprocessed_dataset/test.jsonl`
   - `preprocessed_dataset/frames/...`

#### Key internal configuration

At the top of the script:

- `ACCESS_TOKEN`: Vimeo token
- `EXCEL_PATH`: input Excel path
- `N_CLIPS_POR_VIDEO`, `N_FRAMES_POR_CLIP`, `CLIP_DURATION`, `RESOLUCION_MAX`
- `PCT_TRAIN`: train/test split ratio
- `PROMPT_USER`: prompt appended to each sample

#### Run

```bash
python preprocessing/convert_songs.py
```

#### Expected output

A `preprocessed_dataset/` directory containing:

- `frames/` (PNG files grouped by video)
- `train.jsonl`
- `test.jsonl`

---

## `/train`

### `train_qwen.py`
Fine-tuning script using Unsloth + TRL.

#### What it does

1. Loads a base VLM (`FastVisionModel.from_pretrained`).
2. Applies LoRA (`get_peft_model`).
3. Loads JSONL data with `datasets`.
4. Creates train/eval splits.
5. Trains with `SFTTrainer`.
6. Saves:
   - model/checkpoint artifacts in `output_dir`
   - tokenizer
   - `train_stats.json`
   - LoRA adapters (`Qwen3_Lora`) and merged fp16 weights (`Qwen3_Lora_float16`)

#### Main CLI arguments

- `--dataset_path` (required)
- `--output_dir` (required)
- `--model_name`
- `--train_frac`, `--eval_frac`
- `--max_seq_length`
- training hyperparameters (`--learning_rate`, `--num_train_epochs`, batch size, grad accumulation)
- LoRA options (`--lora_r`, `--lora_alpha`, `--lora_dropout`)

#### Example

```bash
python train/train_qwen.py \
  --dataset_path preprocessing/preprocessed_dataset/train.jsonl \
  --output_dir outputs/qwen3_vl_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2
```

### `Qwen3_VL_(8B)_Vision.ipynb`
Reference notebook (31 cells) for interactive experiments with Unsloth/Qwen3-VL training settings.

---

## `/test`

### `test_model.py`
Batch inference pipeline with Excel export.

#### Workflow

1. Reads an input Excel file with Vimeo links.
2. Downloads each video and trims it (`ffmpeg`, `MAX_SECONDS`).
3. Runs inference for multiple prompts (`PROMPTS`).
4. Saves per-video outputs into `OUTPUT_EXCEL`.

#### Internal config

- `ACCESS_TOKEN`
- `INPUT_EXCEL`, `SHEET_NAME`, `OUTPUT_EXCEL`
- `MODEL_NAME`
- `MAX_SECONDS`

> Note: this script currently uses absolute paths and may require adaptation to your environment.

### `test_unsloth.py`
Alternative folder-based inference script with prompt selection.

- Input: `--input_dir`
- Output: `--output_dir` (one `.txt` file per video)
- Parameters: `--prompt`, `--max_new_tokens`, `--model_name`

> Important: in its current state, this file contains mixed experimental/incomplete blocks and may require cleanup before production use.

### `limpiar_salida.py`
Post-processing for raw model outputs in Excel.

#### What it does

- Reads `prompt_0`, `prompt_1`, `prompt_2` columns.
- Cleans assistant output text.
- Extracts structured fields using regex:
  - `Tipo`
  - `Instrumentos`
  - `Personas`
  - `Ambiente`
- Writes new columns (`tipo_i`, `instrumentos_i`, `personas_i`, `ambiente_i`) to a new Excel file.

#### Run

```bash
python test/limpiar_salida.py
```

### `evaluacion_jerarquica.py`
Evaluation script for predictions vs ground truth labels.

#### Metrics computed

- **Type/category** (single-label): accuracy, precision/recall/F1, Cohen’s kappa.
- **Instruments** (multi-label): subset accuracy, hamming loss, jaccard, precision/recall/F1 (micro/macro/weighted/samples), macro kappa over labels.

#### Dictionary resources

- `variant_to_id.json`: instrument text variants -> canonical IDs.
- `id_to_families.json`: canonical IDs -> hierarchical families.
- `category_map.json`: category normalization map.

### `variant_to_id.json`
Normalization mapping for instrument names (synonyms/orthographic variants).

### `id_to_families.json`
Instrument taxonomy: each canonical ID is associated with one or more families (e.g., `cordofone`, `aerofone`, `tradicional_portugues`).

### `category_map.json`
Normalization map for type/category labels (e.g., narrative-song variants -> `musica narrativa`).

---

## 5) Recommended end-to-end workflow

1. **Prepare environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure data and Vimeo token**
   - Verify required columns in `dataset/Base_pruebas.xlsx`.
   - Update token/paths in `preprocessing/convert_songs.py`.

3. **Generate JSONL dataset + frames**
   ```bash
   python preprocessing/convert_songs.py
   ```

4. **Train LoRA model**
   ```bash
   python train/train_qwen.py --dataset_path preprocessed_dataset/train.jsonl --output_dir outputs/qwen_lora
   ```

5. **Run inference**
   - Batch from Excel: `python test/test_model.py`
   - Folder-based: `python test/test_unsloth.py ...`

6. **Clean outputs**
   ```bash
   python test/limpiar_salida.py
   ```

7. **Evaluate metrics**
   ```bash
   python test/evaluacion_jerarquica.py
   ```

---

## 6) Associated papers


---

## 7) Recommended improvements

- Move hardcoded tokens and paths to environment variables or a `.env` file.
- Add robust path/column validation in all scripts.
- Expose prompts and paths via CLI arguments (especially in `test/`).
- Add deterministic seeding and structured logging for better reproducibility.
- Refactor/clean `test/test_unsloth.py` before operational use.

---

## 8) Troubleshooting

- **`ModuleNotFoundError`**: install dependencies with `pip install -r requirements.txt`.
- **Vimeo API errors**: verify access token and account permissions.
- **GPU OOM**: use 4-bit loading, reduce `max_seq_length`, lower batch size, or reduce frame count.
- **Mismatched Excel files**: verify required columns and ensure row alignment between predictions and ground truth.

---

## 9) License and data usage

No explicit license file is currently included. Add a `LICENSE` file to define reuse/distribution terms. Also verify dataset and video usage rights before training or distributing derived models.

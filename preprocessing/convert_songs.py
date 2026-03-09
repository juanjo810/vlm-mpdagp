"""
Prepara dataset Qwen3-VL desde Excel:
- Descarga Vimeo
- Extrae 2 clips x 8 frames (16 PNG) a 448x448
- Escribe train.jsonl y test.jsonl en formato {"messages":[...]}
"""

import json
import pandas as pd
from pathlib import Path
import requests
import re
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ================= CONFIG =================

# API Vimeo
ACCESS_TOKEN = "d8cd6566978abeabbe2eef523f7d3c11"  # Vimeo token

# Paths
EXCEL_PATH = "../dataset/Base_pruebas.xlsx"
BASE_DIR = Path("preprocessed_dataset")
FRAMES_DIR = BASE_DIR / "frames"
DATASET_TRAIN_JSONL = BASE_DIR / "train.jsonl"
DATASET_TEST_JSONL  = BASE_DIR / "test.jsonl"

# Parameters
N_CLIPS_POR_VIDEO = 2
CLIP_DURATION = 5
N_FRAMES_POR_CLIP = 8
RESOLUCION_MAX = 448

# Split
PCT_TRAIN = 0.8  # resto => test

# Prompt
PROMPT_USER = """ Estos vídeos muestran vídeos tradicionales portugueses (folclore regional) que pueden tratar sobre canciones populares,
historias de vida de personas, poemas recitados entre otras
Instrucciones importantes:
- Describe únicamente lo que sea visible en el vídeo.
- No inventes instrumentos ni elementos no observables.
Siendo música tradicional portuguesa pueden aparecer instrumentos tradicionales portugueses aunque también pueden aparecer
instrumentos convencionales."""
# =========================================

VIDEO_ID_RE = re.compile(r"vimeo\.com/(\d+)")

def setup_dirs():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for f in BASE_DIR.glob("*.jsonl"):
        f.unlink()

# ═══════════════════════════════════════════════════════════════
# VIMEO CODE
# ═══════════════════════════════════════════════════════════════

def extract_vimeo_id(url: str):
    """Extract ID from Vimeo URL"""
    m = VIDEO_ID_RE.search(str(url))
    return m.group(1) if m else None

def get_vimeo_download_url(video_id: str):
    """API Vimeo → download link los quality"""
    if not ACCESS_TOKEN:
        raise RuntimeError("ACCESS_TOKEN vacío. Añade tu token de Vimeo.")
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    r = requests.get(f"https://api.vimeo.com/videos/{video_id}", headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "download" not in data or not data["download"]:
        raise RuntimeError(f"Vimeo API: no download links for video {video_id}")

    # Low quality to save storage/time
    download_options = sorted(data["download"], key=lambda x: x.get("height", 10**9))
    return download_options[0]["link"]

def download_video(url: str, path: Path):
    """Download streaming video"""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

# ═══════════════════════════════════════════════════════════════

def seleccionar_clips_optimos(video_path: Path):
    """Traditional music clips"""
    video = VideoFileClip(str(video_path))
    try:
        duration = float(video.duration)
    finally:
        video.close()

    # Heuristic used:
    # 1. Avoid intros (0-10s)
    # 2. Priorize 20%-80% (central video)
    # 3. 2 clips: "establishment" + "climax"

    clip_starts = [
        max(10.0, duration * 0.25), # 25% → instruments
        max(20.0, duration * 0.60), # 60% → climax
    ]
    safe = [min(start, max(0.0, duration - CLIP_DURATION)) for start in clip_starts]
    return safe

def frames_uniformes(n_frames: int, start_time: float, clip_duration: float):
    """
    Uniform frames in the clip [start_time, start_time+clip_duration].
    """
    if n_frames <= 1:
        return [float(start_time)]
    times = np.linspace(start_time, start_time + clip_duration, n_frames, endpoint=False)
    return [float(t) for t in times]

def parse_instrumentos(raw):
    """
    Normalizing instrument list
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw)
    parts = re.split(r"[;,/]+", s)
    inst = []
    for p in parts:
        p = p.strip()
        if p:
            inst.append(p.lower())
    seen, out = set(), []
    for x in inst:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def build_example(frames_paths, prompt_user, categorias, instrumentos_list):
    """
    Building the example using the following structure:
    messages: [
      {"role": "user", "content": {"type": "image", "image": path_to_image_1}
                                  {"type": "image", "image": path_to_image_2}
                                  {...}
                                  {"type": "text", "text": prompt}
      }
      {"role": "assistant", "content": {"type": "text", "text": "Tipo: tipo\nInstrumentos:instrumentos}}
    ]
    """
    user_content = [{"type": "image", "image": p} for p in frames_paths]
    user_content.append({"type": "text", "text": prompt_user})

    assistant_text = f"Tipo: {categorias}\nInstrumentos: {', '.join(instrumentos_list)}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
    }

def procesar_video(row, idx_row: int):
    """1 row → multiple frame examples"""
    link = row.get("Link", "")
    categorias = row.get("Categorias", "")
    instrumentos_list = parse_instrumentos(row.get("Instrumentos", ""))

    vid_dir = FRAMES_DIR / f"vid_{idx_row:06d}"
    vid_dir.mkdir(parents=True, exist_ok=True)

    vimeo_id = extract_vimeo_id(link)
    if not vimeo_id:
        print(f"❌ Skip {idx_row}: URL inválida: {link}")
        return []

    temp_mp4 = vid_dir / "temp_full.mp4"
    try:
        if not temp_mp4.exists():
            dl_url = get_vimeo_download_url(vimeo_id)
            download_video(dl_url, temp_mp4)

        clip_starts = seleccionar_clips_optimos(temp_mp4)[:N_CLIPS_POR_VIDEO]

        ejemplos = []
        video = VideoFileClip(str(temp_mp4))
        try:
            duration = float(video.duration)

            all_frames = []
            for clip_i, start in enumerate(clip_starts):
                times = frames_uniformes(N_FRAMES_POR_CLIP, start, CLIP_DURATION)
                for frame_i, t in enumerate(times):
                    t = min(max(0.0, t), max(0.0, duration - 1e-3))
                    frame = video.get_frame(t)
                    img = Image.fromarray(frame).resize(
                        (RESOLUCION_MAX, RESOLUCION_MAX),
                        Image.Resampling.LANCZOS,
                    )
                    frame_path = vid_dir / f"clip_{clip_i}_frame_{frame_i:03d}.png"
                    img.save(frame_path)
                    all_frames.append(str(frame_path).replace("\\", "/"))

            expected = N_CLIPS_POR_VIDEO * N_FRAMES_POR_CLIP
            if len(all_frames) != expected:
                if len(all_frames) == 0:
                    print(f"❌ Skip {idx_row}: no se extrajo ningún frame")
                    return []
                if len(all_frames) < expected:
                    all_frames = all_frames + [all_frames[-1]] * (expected - len(all_frames))
                else:
                    all_frames = all_frames[:expected]

            ejemplos.append(build_example(all_frames, PROMPT_USER, categorias, instrumentos_list))
            return ejemplos

        finally:
            video.close()

    except Exception as e:
        print(f"❌ Error procesando {idx_row} (vimeo_id={vimeo_id}): {e}")
        return []

    finally:
        if temp_mp4.exists():
            try:
                temp_mp4.unlink()
            except Exception:
                pass

def write_jsonl(df, out_path: Path, split_name: str):
    """Write to JSONL output"""
    n_examples = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Procesando {split_name}"):
            ejemplos = procesar_video(row, int(i))
            for ej in ejemplos:
                f.write(json.dumps(ej, ensure_ascii=False) + "\n")
                n_examples += 1
    return n_examples

def main():
    setup_dirs()

    df = pd.read_excel(EXCEL_PATH)
    print(f"📊 {len(df)} loaded videos")
    req = {"Link", "Categorias", "Instrumentos"}
    if not req.issubset(df.columns):
        raise ValueError(f"Excel must contain these cols: {sorted(req)}. It has: {list(df.columns)}")

    n_train = int(len(df) * PCT_TRAIN)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df  = df.iloc[n_train:].reset_index(drop=True)

    print(f"🧠 TRAIN videos: {len(train_df)} | 🧪 TEST videos: {len(test_df)}")

    n_train_ex = write_jsonl(train_df, DATASET_TRAIN_JSONL, "TRAIN")
    n_test_ex  = write_jsonl(test_df,  DATASET_TEST_JSONL,  "TEST")

    print(f"✅ train.jsonl: {DATASET_TRAIN_JSONL} (ejemplos: {n_train_ex})")
    print(f"✅ test.jsonl:  {DATASET_TEST_JSONL} (ejemplos: {n_test_ex})")

    n_png = len(list(FRAMES_DIR.rglob("*.png")))
    size_mb = sum(p.stat().st_size for p in FRAMES_DIR.rglob("*.png")) / 1e6
    print(f"🖼️ Total frames: {n_png}")
    print(f"💾 Size frames: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()

import os
import re
import torch
import requests
import pandas as pd
import tempfile
import subprocess
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from unsloth import FastVisionModel

# =========================
# CONFIG
# =========================

ACCESS_TOKEN = "d8cd6566978abeabbe2eef523f7d3c11"
INPUT_EXCEL = "/home/jovyan/projects/VLMs_tests/MUESTREO_BASE_SERPINS.xlsx"
OUTPUT_EXCEL = "resultado_qwen_finetuned.xlsx"
SHEET_NAME = "vimeo"
MAX_SECONDS = 120
MODEL_NAME = "Qwen3_Lora_MPDAGDP"

PROMPTS = [
    """
    Describe el vídeo completo. Identifica SOLO los instrumentos musicales que aparecen siendo tocados, 
    las personas presentes y sus acciones, el tipo exacto de contenido (canción, baile, entrevista, etc.) y el ambiente general.
    """,
    """
    Estos vídeos muestran vídeos tradicionales portugueses (folclore regional) que pueden tratar sobre canciones populares,
    historias de vida de personas, poemas recitados entre otras
        Instrucciones importantes:
        - Describe únicamente lo que sea visible en el vídeo.
        - No inventes instrumentos ni elementos no observables.
    Siendo música tradicional portuguesa pueden aparecer instrumentos tradicionales portugueses aunque también pueden aparecer
    instrumentos convencionales.
    """,
    """
    Vídeos de música tradicional portuguesa para evaluación automática. Evalúa el tipo de vídeo, los instrumentos que aparecen, el ambiente en el
    que se desarrolla y las personas y roles que hacen.
    Las categorías posibles para “Tipo” son estas (elige UNA sola y escribe el nombre EXACTO):
        Agricultura, Artesanato, Dança, Entrevista, Filme / Documentário, Gastronomia,
        História, História de vida, Jogo, Medicina popular, Música de trabalho, Música infantil,
        Música instrumental, Música narrativa, Música para dança, Música sacro-profana,
        Música secular / festiva / comunidade, Oração, Paisagem sonora, Poesia, Poeta,
        Prosa, Religião, Tradição oral.
    Instrumentos pueden aparecer algunos como voz, guitarra, viola braguesa, viola amarantina, cavaquinho, acordeão, gaita de foles, ferrinhos, bombo, adufe, pandeireta, violino.
    Para esto mira el vídeo completo, nombra solo los instrumentos que ves siendo tocados y no inventes instrumentos que no aparezcan claramente.
    """

]

# =========================
# VIMEO
# =========================

def extract_vimeo_id(url):
    return re.search(r"vimeo\.com/(\d+)", url).group(1)

def get_vimeo_download_url(video_id):
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    r = requests.get(f"https://api.vimeo.com/videos/{video_id}", headers=headers)
    r.raise_for_status()
    data = r.json()

    download_options = sorted(data["download"], key=lambda x: x["height"])
    return download_options[0]["link"]

def download_video(url, path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def trim_video(input_path, output_path, seconds):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-t", str(seconds),
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =========================
# QWEN
# =========================

def run_qwen(video_path, prompt_text, processor, model):

    messages = [
        {
            "role": "system",
            "content": [ 
                {
                    "type": "text",
                    "text": """Eres un experto en musicología portuguesa.
                    Analiza el vídeo completo pero devuelve SOLO una descripción final en español con esta estructura exacta:
                        Tipo: [tipo de vídeo como canción narrativa, religiosa, historia de vida, poema recitado, folclore instrumental, entrevista, etc.]
                        Instrumentos: [lista de instrumentos musicales que ves tocando, en singular y en minúsculas]
                        Personas: [número aproximado, roles y acciones]
                        Ambiente: [lugar, contexto, ambiente emocional]
                    No inventes instrumentos ni categorías que no se vean claramente. Si dudas, usa nombres más genéricos (por ejemplo “guitarra”, “percusión”, “viento”). No incluyas razonamientos intermedios, timestamps ni formato markdown."""

                }
            ]
                    #Devuelve SOLO un JSON con '
                    #'{"title","composer"} usando strings o null si no está visible. '
                    #'Ejemplo: {"title":"Mes de mayo","composer":null}. Nada más.'
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    images, videos, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    if videos:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        **video_kwargs,
    )

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
      output_ids = model.generate(
          **inputs,
          max_new_tokens=500,
          use_cache=True,
      )

    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

# =========================
# MAIN
# =========================

def main():

    df = pd.read_excel(INPUT_EXCEL, sheet_name=SHEET_NAME)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = MODEL_NAME, # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        url = row["Link"]
        video_id = extract_vimeo_id(url)

        raw_video = None
        trimmed_video = None

        try:
            download_url = get_vimeo_download_url(video_id)

            raw_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            trimmed_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

            download_video(download_url, raw_video)
            trim_video(raw_video, trimmed_video, MAX_SECONDS)

            responses = []

            for prompt in PROMPTS:
                r = run_qwen(trimmed_video, prompt, processor, model)
                responses.append(r)

            results.append({
                "video_id": video_id,
                "prompt_0": responses[0],
                "prompt_1": responses[1],
                "prompt_2": responses[2],
            })

        except Exception as e:
            print(f"Error en {video_id}: {e}")

        finally:
            for f in [raw_video, trimmed_video]:
                if f and os.path.exists(f):
                    os.remove(f)

    pd.DataFrame(results).to_excel(OUTPUT_EXCEL, index=False)

if __name__ == "__main__":
    main()

# Requisitos (ejecuta antes): 
# pip install -U torch transformers pillow decord

import os
import argparse
from pathlib import Path
import re
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # del repo oficial de Qwen3-VL
from unsloth import FastVisionModel, get_chat_template # FastLanguageModel for LLMs
import torch
from transformers import TextStreamer

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
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




def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS

def sanitize_filename(name: str) -> str:
    # Limpia el nombre para crear el .txt
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL — Narrativa para todos los vídeos de un directorio")
    parser.add_argument("--input_dir", type=str, required=True, help="Directorio con vídeos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio de salida para los .txt")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Límite de tokens generados")
    parser.add_argument("--prompt", type=int, default=1,help="Tipo de instrucción para el modelo (0=prompt básico, 1=prompt contexto medio, 2=contexto amplio)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Nombre del checkpoint en HF")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Selección del prompt
    prompt_index = args.prompt
    if 0 <= prompt_index < len(PROMPTS):
        selected_prompt = PROMPTS[prompt_index]
    else:
        raise ValueError(f"Índice de prompt inválido: {prompt_index}")
    print("Has seleccionado el siguiente prompt:")
    print(selected_prompt)

    # Carga de procesador y modelo (clase recomendada por Transformers para VLMs)
    
    model, processor = FastVisionModel.from_pretrained(
        "unsloth/gemma-3-4b-pt",
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    processor = get_chat_template(
        processor,
        "gemma-3"
    )

    FastVisionModel.for_inference(model)  # Enable for inference!

    # Recorre todos los vídeos del directorio
    video_paths = [p for p in in_dir.iterdir() if is_video_file(p)]
    if not video_paths:
        print(f"No se encontraron vídeos en: {in_dir}")
        return

    for vid_path in sorted(video_paths):
        try:
            # Construcción del mensaje multimodal para cada vídeo
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
                        {"type": "video", "video": str(vid_path)},
                        {"type": "text", "text": selected_prompt},
                    ],
                }
            ]

    
    input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")


    text_streamer = TextStreamer(processor, skip_prompt = True)
    result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                            use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)
            # Procesa frames/kwargs del vídeo según utilidades de Qwen
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
            if videos is not None and len(videos) > 0:
                videos, video_metadatas = zip(*videos)  # tuples -> two tuples
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            # Aplica plantilla de chat y empaqueta multimodal con el processor
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[prompt],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                **video_kwargs,
            )

            # Mueve tensores al dispositivo del modelo
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            # Generación
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Guarda un .txt por vídeo
            stem = sanitize_filename(vid_path.stem)
            out_file = out_dir / f"{stem}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text + "\n")

            print(f"OK: {vid_path.name} -> {out_file.name}")
        except Exception as e:
            print(f"ERROR en {vid_path.name}: {e}")

if __name__ == "__main__":
    main()


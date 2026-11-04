import os
import re
from typing import List
import pandas as pd
from pydantic import BaseModel


class SceneChunk(BaseModel):
  film_title: str
  scene_header: str
  content: str


def clean_text(text: str) -> str:
  text = text.replace("\r\n", "\n")
  text = re.sub(r"\n[ \t]*\d+\.[ \t]*(?=\n)", "", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text



def split_large_scene(
    scene_text: str, max_chars: int = 700, overlap: int = 150
) -> List[str]:
  """Divide escenas largas en sub-bloques con solapamiento (overlap)

  para no exceder el contexto del modelo de embeddings ni perder significado.
  """
  if len(scene_text) <= max_chars:
    return [scene_text]

  sub_chunks = []
  start = 0
  text_length = len(scene_text)

  while start < text_length:
    end = start + max_chars
    chunk = scene_text[start:end]

    # Intentamos ajustar el corte al final de una frase o línea si no estamos al final del texto
    if end < text_length:
      last_newline = chunk.rfind("\n")
      if last_newline > max_chars // 2:
        end = start + last_newline + 1
        chunk = scene_text[start:end]

    sub_chunks.append(chunk.strip())


    if end >= text_length:
      break


    start = max(end - overlap, start + 1)

  return [c for c in sub_chunks if len(c) > 50]


def parse_script_into_scenes(file_path: str) -> List[SceneChunk]:
  """Lee un guion en .txt y lo divide en escenas usando expresiones regulares,

  aplicando sub-chunking en escenas que excedan el umbral de caracteres.
  """
  file_name = os.path.basename(file_path)
  film_title = (
      os.path.splitext(file_name)[0].replace("_", " ").strip(". ").title()
  )

  with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

  cleaned_text = clean_text(raw_text)

  # Patrón Regex flexible para detectar encabezados de escena
  scene_pattern = r"(?im)^(?=[ \t]*(?:INT|EXT|INT[\./]EXT|EXT[\./]INT)\b\.?[ \t]+)"
  raw_scenes = re.split(scene_pattern, cleaned_text)

  chunks = []
  for scene in raw_scenes:
    scene = scene.strip()
    if not scene:
      continue

    lines = [line.strip() for line in scene.split("\n") if line.strip()]
    header = lines[0] if lines else "UNKNOWN SCENE"

    if len(scene) > 50:
      # Aplicamos sub-chunking para escenas extensas
      sub_scenes = split_large_scene(scene, max_chars=700, overlap=150)

      for sub_idx, sub_content in enumerate(sub_scenes):
        # Mantenemos el header original o le añadimos un índice si fue dividida
        part_header = (
            f"{header} (Part {sub_idx + 1})" if len(sub_scenes) > 1 else header
        )

        if sub_idx > 0:
          sub_content = f"{header}\n{sub_content}"

        chunks.append(
            SceneChunk(
                film_title=film_title,
                scene_header=part_header,
                content=sub_content,
            )
        )

  return chunks


if __name__ == "__main__":
  raw_dir = "data/raw"
  processed_dir = "data/processed"
  os.makedirs(processed_dir, exist_ok=True)

  files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
  all_scenes = []

  print(f"Procesando {len(files)} guiones encontrados en {raw_dir}...\n")

  for file_name in files:
    file_path = os.path.join(raw_dir, file_name)
    scenes = parse_script_into_scenes(file_path)
    all_scenes.extend([s.model_dump() for s in scenes])
    print(f"  - {file_name}: {len(scenes)} chunks de escena extraídos.")

  df = pd.DataFrame(all_scenes)

  print("\n--- RESUMEN CON PANDAS ---")
  print(f"Total de chunks extraídos: {len(df)}")
  print("\nChunks por película:")
  print(df["film_title"].value_counts())

  # Guardamos el dataset procesado
  output_csv = os.path.join(processed_dir, "scenes_dataset.csv")
  df.to_csv(output_csv, index=False, encoding="utf-8")
  print(f"\nDataset guardado con éxito en: {output_csv}")

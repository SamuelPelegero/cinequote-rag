import os
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd


def build_vector_database(
    csv_path: str = "data/processed/scenes_dataset.csv",
    db_path: str = "data/chroma_db",
    collection_name: str = "cine_scenes",
):
  """Lee el CSV de escenas, genera embeddings y los guarda en ChromaDB."""

  if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"No se encontró el dataset en {csv_path}. Ejecuta parser.py primero."
    )

  print(f"Cargando dataset desde {csv_path}...")
  df = pd.read_csv(csv_path)

  # 1. Configurar la función de embeddings (Modelo ligero y rápido de Hugging Face)
  # 'all-MiniLM-L6-v2' convierte cada texto en un vector de 384 dimensiones
  print("Cargando modelo de embeddings (Sentence-Transformers)...")
  ef = embedding_functions.SentenceTransformerEmbeddingFunction(
      model_name="all-MiniLM-L6-v2"
  )

  # 2. Inicializar la base de datos ChromaDB con persistencia en disco
  print(f"Inicializando ChromaDB en {db_path}...")
  client = chromadb.PersistentClient(path=db_path)

  # Creamos o recuperamos la colección asignándole la función de embedding
  collection = client.get_or_create_collection(
      name=collection_name, embedding_function=ef
  )

  # 3. Preparar los datos para ChromaDB
  # Necesitamos 4 listas del mismo tamaño: ids, documents (texto), metadatas (dict)
  documents = df["content"].tolist()
  metadatas = [
      {"film_title": row["film_title"], "scene_header": row["scene_header"]}
      for _, row in df.iterrows()
  ]
  # Creamos IDs únicos para cada escena, ej: scene_0, scene_1...
  ids = [f"scene_{i}" for i in range(len(df))]

  print(f"Insertando {len(documents)} escenas en la base de datos vectorial...")

  # Insertar por lotes (batches) para optimizar memoria
  batch_size = 100
  for i in range(0, len(documents), batch_size):
    end_i = i + batch_size
    collection.add(
        ids=ids[i:end_i],
        documents=documents[i:end_i],
        metadatas=metadatas[i:end_i],
    )
    print(f"  - Lote {i // batch_size + 1} insertado ({min(end_i, len(documents))}/{len(documents)})")

  print(
      f"\n ¡Éxito! Base de datos vectorial construida con {collection.count()}"
      " escenas."
  )


if __name__ == "__main__":
  build_vector_database()
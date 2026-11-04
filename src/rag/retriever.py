import chromadb
from chromadb.utils import embedding_functions


class ScriptRetriever:

  def __init__(
      self,
      db_path: str = "data/chroma_db",
      collection_name: str = "cine_scenes",
  ):
    # Cargamos la misma función de embeddings para mapear la consulta al mismo espacio vectorial
    self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    self.client = chromadb.PersistentClient(path=db_path)
    self.collection = self.client.get_collection(
        name=collection_name, embedding_function=self.ef
    )

  def search(self, query: str, top_k: int = 3, film_title: str = None):
    """Busca las K escenas más afines semánticamente a la consulta del usuario, con opción a filtrar por película."""
    # Permitimos filtrado por metadatos (ej. buscar solo dentro de El Padrino)
    where_clause = {"film_title": film_title} if film_title else None

    results = self.collection.query(
        query_texts=[query], n_results=top_k, where=where_clause
    )

    retrieved_scenes = []
    for i in range(len(results["ids"][0])):
      scene_data = {
          "film_title": results["metadatas"][0][i]["film_title"],
          "scene_header": results["metadatas"][0][i]["scene_header"],
          "content": results["documents"][0][i],
          "distance": results["distances"][0][i]
          if "distances" in results
          else None,
      }
      retrieved_scenes.append(scene_data)

    return retrieved_scenes


if __name__ == "__main__":
  retriever = ScriptRetriever()

  # Consulta de prueba por concepto semántico
  query_test = "A conversation about loyalty, money or revenge"
  print(f"Buscando escenas relacionadas con: '{query_test}'...\n")

  results = retriever.search(query=query_test, top_k=2)

  for i, res in enumerate(results, 1):
    print(f"--- RESULTADO {i} ---")
    print(f"Película: {res['film_title']}")
    print(f"Escena: {res['scene_header']}")
    print(f"Distancia vectorial: {res['distance']:.4f}")
    print(f"Texto (primeros 250 caracteres):\n{res['content'][:250]}...\n")
from typing import List, Optional
from pydantic import BaseModel, Field


# Esquema para la petición del cliente (Request)
class SearchQueryRequest(BaseModel):
  query: str = Field(
      ...,
      description="Frase, diálogo o concepto semántico a buscar en los guiones",
      example="A conversation about loyalty, money or revenge",
  )
  top_k: int = Field(
      default=3,
      ge=1,
      le=10,
      description="Número de escenas a recuperar (entre 1 y 10)",
  )
  film_title: Optional[str] = Field(
      default=None,
      description="Filtro opcional para buscar solo en una película concreta",
      example="El Padrino",
  )


# Esquema para cada escena devuelta en la lista de resultados
class SceneResponse(BaseModel):
  film_title: str
  scene_header: str
  content: str
  distance: Optional[float] = None


# Esquema de respuesta global de la API (Response)
class SearchQueryResponse(BaseModel):
  query: str
  total_results: int
  results: List[SceneResponse]

# Esquema para la respuesta RAG (Retrieval + Generation)
class RAGResponse(BaseModel):
  query: str
  generated_answer: str
  retrieved_context: List[SceneResponse]
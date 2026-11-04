from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from src.api.schemas import (
    RAGResponse,
    SceneResponse,
    SearchQueryRequest,
    SearchQueryResponse,
)
from src.rag.generator import RAGGenerator
from src.rag.retriever import ScriptRetriever

retriever = None
generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  global retriever, generator
  print("Inicializando ScriptRetriever y RAGGenerator...")
  retriever = ScriptRetriever()
  generator = RAGGenerator()
  print("¡Retriever y Generator listos!")
  yield
  print("Cerrando servicio API...")


app = FastAPI(
    title="CineQuote RAG API",
    description=(
        "API REST para búsqueda semántica y generación RAG sobre guiones"
        " cinematográficos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Health Check"])
def health_check():
  return {"status": "ok", "service": "CineQuote RAG API active"}


@app.post(
    "/api/v1/search", response_model=SearchQueryResponse, tags=["Retrieval"]
)
def search_scenes(payload: SearchQueryRequest):
  """Devuelve solo las escenas recuperadas por la base de datos vectorial (Retrieval puro)."""
  if not retriever:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="El motor de búsqueda no está inicializado.",
    )

  try:
    raw_results = retriever.search(
        query=payload.query,
        top_k=payload.top_k,
        film_title=payload.film_title,
    )

    formatted_scenes = [SceneResponse(**res) for res in raw_results]

    return SearchQueryResponse(
        query=payload.query,
        total_results=len(formatted_scenes),
        results=formatted_scenes,
    )
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Error durante la búsqueda vectorial: {str(e)}",
    )


@app.post("/api/v1/rag", response_model=RAGResponse, tags=["RAG (Generation)"])
def generate_rag_answer(payload: SearchQueryRequest):
  """Recupera contexto relevante y genera una respuesta razonada usando un LLM (RAG completo)."""
  if not retriever or not generator:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="El servicio RAG no está totalmente inicializado.",
    )

  try:
    # 1. Fase de Retrieval
    raw_results = retriever.search(
        query=payload.query,
        top_k=payload.top_k,
        film_title=payload.film_title,
    )

    # 2. Fase de Generation
    answer = generator.generate_answer(
        query=payload.query, retrieved_scenes=raw_results
    )

    formatted_scenes = [SceneResponse(**res) for res in raw_results]

    return RAGResponse(
        query=payload.query,
        generated_answer=answer,
        retrieved_context=formatted_scenes,
    )
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Error durante el proceso RAG: {str(e)}",
    )
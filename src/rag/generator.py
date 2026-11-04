import os
from typing import List
from openai import OpenAI


class RAGGenerator:

  def __init__(self, base_url: str = None, api_key: str = None):
    self.base_url = base_url or os.getenv(
        "LLM_BASE_URL", "http://localhost:11434/v1"
    )
    self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")

    self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

  def generate_answer(
      self,
      query: str,
      retrieved_scenes: List[dict],
      model: str = "qwen2.5:3b",  
  ) -> str:
    """Construye el prompt con las escenas de ChromaDB y consulta a Ollama en local."""
    context_blocks = []
    for idx, scene in enumerate(retrieved_scenes, 1):
      block = (
          f"--- ESCENA {idx} ---\n"
          f"Película: {scene['film_title']}\n"
          f"Encabezado: {scene['scene_header']}\n"
          f"Contenido:\n{scene['content']}\n"
      )
      context_blocks.append(block)

    context_str = "\n".join(context_blocks)

    system_prompt = (
        "Eres un analista cinematográfico experto y un asistente de RAG."
        " Tu tarea es responder a la duda del usuario basándote ÚNICAMENTE en"
        " las escenas de guiones proporcionadas como contexto.\n"
        "Si la información no está en el contexto, indica amablemente que no"
        " dispones de esa escena en la base de datos.\n"
        "Mantén un tono profesional, claro y cita el título de la película y la"
        " escena al responder."
    )

    user_prompt = f"""CONTEXTO RECUPERADO:
{context_str}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""

    try:
      response = self.client.chat.completions.create(
          model=model,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.3,
      )
      return response.choices[0].message.content
    except Exception as e:
      return (
          f"Error al conectar con Ollama local ({self.base_url})."
          f" Detalles: {str(e)}"
      )
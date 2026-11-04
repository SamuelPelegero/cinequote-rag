# CineQuote RAG

Pipeline RAG (*Retrieval-Augmented Generation*) de extremo a extremo para la ingesta, la búsqueda semántica y la generación de respuestas sobre guiones cinematográficos no estructurados. Todo se ejecuta en local.

## Arquitectura

```
Fase offline (una vez)
  guiones .txt → parser (chunks + metadatos) → CSV → embeddings → ChromaDB

Fase online (cada consulta)
  pregunta → embedding → ChromaDB (top-k) → prompt con contexto → Qwen 2.5 (Ollama) → respuesta + fuentes
```

1. **Ingesta y parseo** (`src/ingestion/parser.py`): carga de guiones crudos (`.txt`), limpieza con expresiones regulares, detección de encabezados de escena (`INT.` / `EXT.`) y *sub-chunking* con solapamiento (ventana deslizante de 700 caracteres con 150 de overlap) para no exceder el contexto del modelo de embeddings. Cada chunk lleva metadatos (`film_title`, `scene_header`) y se valida con **Pydantic**. El dataset se explora y se guarda como CSV con **pandas**.
2. **Indexación y recuperación** (`src/ingestion/embedder.py`, `src/rag/retriever.py`): embeddings de 384 dimensiones con **Sentence-Transformers** (`all-MiniLM-L6-v2`, que se ejecuta sobre PyTorch), almacenados en **ChromaDB** con persistencia en disco. Soporta filtrado por metadatos (`film_title`). La métrica de distancia es la de Chroma por defecto (L2); menor distancia significa mayor similitud.
3. **API REST** (`src/api/`): servicio construido con **FastAPI** y esquemas **Pydantic**. Usa `lifespan` para cargar el modelo de embeddings, la base vectorial y el cliente del LLM una sola vez al arrancar.
4. **Generación local** (`src/rag/generator.py`): el contexto recuperado se inyecta en el prompt de **Qwen 2.5 3B**, servido con **Ollama**, que genera la respuesta basándose en las escenas recuperadas. No hay entrenamiento ni *fine-tuning*: se usa un modelo preentrenado.

## Estructura del proyecto

```
cinequote-rag/
├── data/
│   ├── raw/            # guiones .txt (no incluidos en el repo)
│   ├── processed/      # scenes_dataset.csv
│   └── chroma_db/      # base vectorial persistida
├── src/
│   ├── ingestion/      # parser.py, embedder.py
│   ├── rag/            # retriever.py, generator.py
│   └── api/            # main.py, schemas.py
├── requirements.txt
└── README.md
```

## Instalación y uso

### 1. Requisitos previos

- Python 3.11 o superior (desarrollado y probado con 3.13).
- [Ollama](https://ollama.com/) instalado y en ejecución.

### 2. Entorno

```bash
git clone https://github.com/<tu-usuario>/cinequote-rag.git
cd cinequote-rag

python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Datos y modelo

- Coloca tus guiones en formato `.txt` en `data/raw/`. Los guiones no se incluyen en este repositorio por motivos de derechos de autor.
- Descarga el modelo local:

```bash
ollama pull qwen2.5:3b
```

### 4. Ejecución

```bash
python -m src.ingestion.parser      # genera data/processed/scenes_dataset.csv
python -m src.ingestion.embedder    # construye data/chroma_db
uvicorn src.api.main:app --reload   # API en http://127.0.0.1:8000
```

La primera ejecución del embedder descarga el modelo de embeddings (~90 MB) y puede tardar varios minutos.

### 5. Probar

Abre http://127.0.0.1:8000/docs y usa `POST /api/v1/rag` con, por ejemplo:

```json
{
  "query": "What does Sollozzo ask Don Corleone?",
  "top_k": 3,
  "film_title": "El Padrino"
}
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Health check del servicio |
| POST | `/api/v1/search` | Recuperación pura: devuelve las escenas más similares con su distancia |
| POST | `/api/v1/rag` | RAG completo: devuelve la respuesta generada y las escenas usadas como fuentes |

Parámetros de entrada (`/search` y `/rag`): `query` (obligatorio), `top_k` (1–10, por defecto 3) y `film_title` (opcional, coincidencia exacta).

## Configuración

El generador es intercambiable por variables de entorno. Por defecto apunta a Ollama en local:

| Variable | Valor por defecto |
|----------|-------------------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` |
| `LLM_API_KEY` | `ollama` |

Cualquier servidor compatible con la API de OpenAI puede usarse cambiando estas variables y el nombre del modelo.

## Limitaciones y próximos pasos


**Limitaciones conocidas**

- Un modelo de 3B parámetros a veces confunde quién dice qué en diálogos cruzados o mezcla información, aunque el prompt le exija ceñirse al contexto. La recuperación suele ser razonable; el punto débil es la generación.
- El chunking por caracteres puede cortar a mitad de frase, y los guiones traen sangrías que gastan espacio del chunk.
- No hay umbral de relevancia: siempre se devuelven `top_k` escenas, aunque ninguna sea relevante.
- No existe aún una evaluación formal de la calidad de la recuperación.
- El filtro por película requiere coincidencia exacta del título.

**Próximos pasos**

- Conjunto de evaluación (preguntas con la escena esperada) y métrica recall@k.
- Chunking por límites de diálogo y limpieza de sangrías.
- Búsqueda híbrida (BM25 + embeddings) y *reranking* con un cross-encoder.
- Distancia coseno (`hnsw:space`) y umbral de distancia máxima.
- Modelo mayor (por ejemplo `qwen2.5:7b`) y propagación correcta de errores del LLM (502/503).
- Empaquetado con Docker.

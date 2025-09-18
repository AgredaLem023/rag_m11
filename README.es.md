# Pipeline RAG Modular

Sistema RAG (Retrieval-Augmented Generation) modular de extremo a extremo que ingiere corpora multidocumento (directorios JSONL/TXT, archivos JSONL únicos o PDFs), persiste embeddings densos con ChromaDB, construye un índice disperso BM25, realiza recuperación híbrida con Reciprocal Rank Fusion (RRF), reordena con un cross-encoder y genera respuestas fundamentadas con citas usando un LLM.

## Destacados
- Arquitectura modular: ingestión, recuperación, reranking y generación están separadas en módulos enfocados.
- Almacén vectorial persistente: ChromaDB con embeddings de SentenceTransformer.
- Recuperación dispersa: BM25 en memoria con caché en disco.
- Recuperación híbrida: denso + BM25 fusionados vía RRF.
- Reranking: selección con cross-encoder y abstención basada en umbral.
- Respuestas: fundamentadas, con estilo de citas; se abstiene cuando la evidencia es insuficiente.

## Estructura del repositorio (archivos clave)
- [rag.py](rag.py) — Adaptador de CLI compatible hacia atrás que delega al paquete rag/.
- [requirements.txt](requirements.txt) — Dependencias de Python.
- [data/2wikimqa.jsonl](data/2wikimqa.jsonl) — Conjunto de datos de ejemplo.
- [rag/cli.py](rag/cli.py) — Ejecutable de CLI (modos build y query).
- [rag/config.py](rag/config.py) — Configuración inmutable (reemplaza variables globales en tiempo de ejecución).
- [rag/constants.py](rag/constants.py) — Constantes compartidas (p. ej., mensaje de abstención).
- [rag/ingestion/parser.py](rag/ingestion/parser.py) — Segmentador de pasajes 2Wiki + parser genérico de registros.
- [rag/ingestion/chunking.py](rag/ingestion/chunking.py) — Constructor de chunks por oraciones para PDFs.
- [rag/ingestion/persist.py](rag/ingestion/persist.py) — Orquestación de ingestión y persistencia (Chroma + caché BM25).
- [rag/retrieval/bm25.py](rag/retrieval/bm25.py) — BM25, tokenización, cargador de caché.
- [rag/retrieval/expansion.py](rag/retrieval/expansion.py) — Expansión de consultas con LLM (multi-query).
- [rag/retrieval/fusion.py](rag/retrieval/fusion.py) — RRF + orquestación de recuperación híbrida.
- [rag/rerank/cross_encoder.py](rag/rerank/cross_encoder.py) — Reranking con cross-encoder y selección.
- [rag/generation/answer.py](rag/generation/answer.py) — Síntesis de respuestas con citas.

## Prerrequisitos
- Recomendado Python 3.10–3.12.
- Una clave de API de OpenAI para expansión de consultas y generación de respuestas.
- Acceso a Internet en la primera ejecución para descargar modelos y pesos de transformers.

## Configuración
1) Crear y activar un entorno virtual (se muestra Windows PowerShell; adapte a su shell/SO):
   - `python -m venv .venv`
   - `.\\.venv\\Scripts\\Activate.ps1`
2) Instalar dependencias:
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
3) Configurar su clave de API de OpenAI:
   - Windows (PowerShell): `$env:OPENAI_API_KEY="sk-...tu_clave..."`
   - macOS/Linux (bash/zsh): `export OPENAI_API_KEY="sk-...tu_clave..."`
   - O cree un archivo .env en el directorio raíz con `OPENAI_API_KEY=Tu-clave-openai`

## Uso
Construir el índice persistente y la caché BM25 (hágalo una vez por colección):
- `python rag.py --build --path data/2wikimqa.jsonl --collection demo`

Formular una pregunta usando el índice existente:
- `python rag.py --path data/2wikimqa.jsonl --collection demo --query "Where did Helena Carroll's father study?"`

Notas:
- `--path` debe apuntar al mismo corpus usado para construir. Si falta la caché BM25, el sistema la reconstruirá al vuelo.
- Para PDFs, el segmentado a nivel de oración se habilita automáticamente; para pasajes JSONL (p. ej., 2Wiki), cada pasaje es un único chunk.

## Cómo funciona (interacciones entre módulos)

```mermaid
flowchart TB
  subgraph CLI
    CLI[rag/cli.py]
  end

  subgraph Ingestion
    PARSER[rag/ingestion/parser.py]
    CHUNK[rag/ingestion/chunking.py]
    PERSIST[rag/ingestion/persist.py]
  end

  subgraph Retrieval
    BM25[rag/retrieval/bm25.py]
    EXP[rag/retrieval/expansion.py]
    FUSION[rag/retrieval/fusion.py]
  end

  subgraph Rerank
    XENC[rag/rerank/cross_encoder.py]
  end

  subgraph Generation
    ANS[rag/generation/answer.py]
  end

  CLI -->|--build| PERSIST
  CLI -->|query| EXP --> FUSION
  FUSION --> BM25
  FUSION --> XENC
  XENC --> ANS

  PARSER --> PERSIST
  CHUNK --> PERSIST
```

- Modo build:
  - [rag/cli.py](rag/cli.py) llama a [rag/ingestion/persist.py](rag/ingestion/persist.py) para:
    - Parsear/normalizar documentos vía [rag/ingestion/parser.py](rag/ingestion/parser.py)
    - Trocear PDFs por oraciones vía [rag/ingestion/chunking.py](rag/ingestion/chunking.py)
    - Persistir en Chroma y construir BM25, cacheando tokens/doc_ids en un pickle
- Modo query:
  - [rag/cli.py](rag/cli.py) carga Chroma y la caché BM25 desde [rag/retrieval/bm25.py](rag/retrieval/bm25.py)
  - Expande la consulta del usuario vía [rag/retrieval/expansion.py](rag/retrieval/expansion.py)
  - Realiza recuperación densa + BM25 y fusión RRF vía [rag/retrieval/fusion.py](rag/retrieval/fusion.py)
  - Reordena candidatos vía [rag/rerank/cross_encoder.py](rag/rerank/cross_encoder.py)
  - Genera una respuesta fundamentada vía [rag/generation/answer.py](rag/generation/answer.py), usando solo texto de contexto y citas en línea

## Qué se genera
- Directorio persistente de Chroma: `chroma_<collection>/` (p. ej., `chroma_demo/`)
  - Almacena el índice denso (embeddings) y metadatos
- Archivo de caché BM25: `bm25_<collection>.pkl` (p. ej., `bm25_demo.pkl`)
  - Contiene textos de chunks tokenizados y IDs de chunks alineados
- Salida en consola:
  - Respuesta fundamentada con citas entre corchetes en línea
  - Citas seleccionadas y puntuaciones de reranking
  - Desglose de tiempos de ingest, expand, retrieve, rerank, generate

## Solución de problemas
- La primera ejecución es lenta:
  - `sentence-transformers` y `transformers` descargarán modelos a tu caché de HF
  - Chroma inicializará un almacén local persistente bajo `chroma_<collection>/`
- Falta `OPENAI_API_KEY`:
  - La expansión multi-consulta y la generación de respuestas requieren una clave válida. Establece la variable de entorno antes de ejecutar.
- Reconstrucción de índices:
  - Elimina `chroma_<collection>/` y `bm25_<collection>.pkl` para reconstruir por completo; o vuelve a ejecutar con `--build` para refrescar.
- “Sombras” de paquetes en Windows:
  - El adaptador [rag.py](rag.py) garantiza que los imports se resuelvan al paquete `rag/` preservando la UX de `python rag.py`.
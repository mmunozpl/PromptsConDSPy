# Código del capítulo 7 — flujos: RAG, agentes y herramientas

Recuperación y RAG sobre un corpus de arXiv (cs.CR, seguridad) con modelo local
(vLLM) y embeddings locales en GPU. Requiere vLLM en :8000.

## Dependencias

python>=3.10, dspy==3.2.1, vllm==0.24.0, rank_bm25, transformers (embeddings vía
AutoModel; NO se usa sentence-transformers por conflicto con transformers/peft),
numpy, torch. Entorno: conda `envDSPy` (RTX 5090). Modelo LM: Qwen2.5-7B local;
embedder: BAAI/bge-small-en-v1.5 y all-MiniLM-L6-v2 (AutoModel + pooling).

## Datos

`datos_arxiv.py` descarga 400 resúmenes de arXiv cs.CR por la API pública (caché
en `salidas/corpus_arxiv.json`) y genera 100 consultas de **auto-recuperación**:
el modelo local escribe una pregunta por documento; el documento de origen es el
relevante (gold). Relevancia por construcción, sin etiquetado a mano.

## Scripts

- `experimentos.py` — registra las cifras `c7-*` en `../comun/cifras.csv`:
  - `recuperacion` — BM25 (léxica) vs densa vs híbrida (RRF): R@1, R@5, MRR.
  - `troceado` — R@5 por tamaño de fragmento (40/80 palabras vs documento).
  - `embedder` — R@5 de bge-small vs all-MiniLM.
  - `fidelidad` — % de respuestas RAG fundadas en el contexto (juez LM).
  - `rag` — RAG base vs optimizado (BootstrapFewShot), correctas por juez.
  - `agente` — agente ReAct con herramienta de búsqueda vs tubería de un paso.
  - Ejecutar todo: `python experimentos.py`.

## Reproducibilidad

Recuperación (BM25/densa) es determinista. Las métricas con LM (fidelidad, rag,
agente) son estocásticas y varían entre corridas (juez LM local); se reportan
como medición única con esa cautela declarada. ColBERTv2 (multivector) queda
fuera: exige su propio índice/motor. Tras ejecutar, `exportar_cifras.py` +
`verificar_cifras.py`.

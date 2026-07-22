# Código del capítulo 3 — clasificación en un SOC

Experimentos sobre el corpus **CSIC 2010** (peticiones HTTP normales/anómalas)
con un modelo local servido por vLLM. Producen todas las cifras de cabecera del
capítulo y las registran en `../comun/cifras.csv` (nunca se teclean a mano).

## Dependencias

python>=3.10, dspy==3.2.1, vllm==0.24.0, scikit-learn, numpy, datasets.
Entorno de referencia: conda `envDSPy` (RTX 5090, CUDA 13).

## Datos

**CSIC 2010 HTTP** (binaria: benigno/ataque), del espejo público
`bridge4/CSIC2010_dataset_classification` en HuggingFace; `datos_csic.py` lo
descarga con caché y lo particiona de forma estratificada. Licencias y
procedencia, en el apéndice de *datasets* de la obra completa.

## Modelo local (vLLM)

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.85
```
En RTX 5090/CUDA 13 el *sampler* JIT de flashinfer no compila: de ahí
`VLLM_USE_FLASHINFER_SAMPLER=0` y `--enforce-eager` (ver apéndice A).

## Scripts

- `datos_csic.py` — descarga, serializa y particiona el corpus (estratificado,
  semilla fija). Ejecutar solo: `python datos_csic.py` (imprime un resumen).
- `experimentos.py` — mide todo y registra las cifras. Requiere vLLM en :8000.
  - `python experimentos.py` — todos los grupos.
  - `python experimentos.py optimizar baselines confusion ofuscacion asr
    calibracion clasico fraccion` — grupos concretos.
  - Registra: exactitud/F1/coste de Predict, ChainOfThought y compilado (N=3
    corridas sin caché, media ± desviación); matriz de confusión y κ del
    compilado; robustez ante ofuscación e inyección (ASR); calibración (ECE
    bruto vs reescalado); baseline clásico TF-IDF+LogReg; fracción que llega al
    LM; coste de compilación. El artefacto compilado se guarda en
    `salidas/clasificador_compilado.json`.

## Reproducibilidad

Métricas de LM: media ± desviación sobre N=3 corridas con la caché desactivada.
Modelo: Qwen2.5-7B-Instruct local (2026-07); las cifras son de
ese modelo y ese corpus. Tras ejecutar, `python ../../scripts/exportar_cifras.py`
regenera las macros `\res` y `python ../../scripts/verificar_cifras.py` es el
gate.

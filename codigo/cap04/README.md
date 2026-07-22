# Código del capítulo 4 — evaluación rigurosa

Reutiliza el corpus CSIC 2010 y el **clasificador compilado del capítulo 3**
(`../cap03/salidas/clasificador_compilado.json`) para medir las cifras de
evaluación del capítulo. Requiere haber corrido antes `../cap03/experimentos.py`
(para que exista el artefacto compilado) y vLLM en :8000.

## Dependencias

python>=3.10, dspy==3.2.1, vllm==0.24.0, scikit-learn, numpy, datasets.
Entorno: conda `envDSPy` (RTX 5090, CUDA 13). Modelo: Qwen2.5-7B-Instruct local.

## Scripts

- `evaluacion.py` — 11 grupos, registra las cifras `c4-*` en `../comun/cifras.csv`:
  - `tabla` — exactitud, F1 macro, κ de Cohen y MCC del compilado.
  - `acc-media` — exactitud como media ± desviación sobre N corridas.
  - `varianza` — desglose: varianza por decodificación (temp>0) frente a
    muestreo (distintas particiones).
  - `consistencia` — fracción de ejemplos vacilantes (desacuerdo en K muestras).
  - `rebanadas` — F1 por clase y exactitud por longitud de la petición.
  - `vecino` — fuga por vecindad: similitud máxima train-test (TF-IDF) y
    fracción de casi-gemelos.
  - `latencia` — percentiles p50/p95 (pasada secuencial).
  - `umbral` — barrido del umbral óptimo por costes asimétricos.
  - `contaminacion` — gap de exactitud limpio frente a perturbado.
  - `perfil` — reparto de los errores (falsos negativos frente a positivos).
  - `juez` — κ de un juez LM independiente frente a la referencia.
  - Ejecutar todo: `python evaluacion.py`. Grupos sueltos: `python evaluacion.py
    tabla varianza vecino`.

## Reproducibilidad

Métricas de LM: media ± desviación sobre N=3 corridas sin caché.
Modelo Qwen2.5-7B local (2026-07). Tras ejecutar,
`python ../../scripts/exportar_cifras.py` regenera las macros y
`python ../../scripts/verificar_cifras.py` es el gate.

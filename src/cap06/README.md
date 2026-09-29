# Código del capítulo 6 — optimización de instrucciones

Compara los optimizadores de instrucciones sobre CSIC 2010 con el modelo local.
Reutiliza el corpus y la firma del capítulo 3. Requiere vLLM en :8000 y `optuna`
(para MIPROv2: `pip install optuna`).

## Dependencias

python>=3.10, dspy==3.2.1, vllm==0.24.0, optuna, scikit-learn, numpy, datasets.
Entorno: conda `envDSPy` (RTX 5090). Modelo: Qwen2.5-7B-Instruct local.

## Scripts

- `experimentos.py` — registra las cifras `c6-*` en `../comun/cifras.csv`:
  - `baseline` — ChainOfThought sin optimizar.
  - `copro` — COPRO (solo instrucción; breadth=3, depth=2).
  - `mipro` — MIPROv2 (instrucción + demostraciones; auto="light").
  - `gepa` — GEPA (evolución reflexiva; reflection_lm local, 600 llamadas).
  - Para cada uno: exactitud, F1, coste de compilación (Mtok) y de inferencia
    (tok/pred). Ejecutar todo: `python experimentos.py`. Cada programa
    compilado se guarda en `salidas/<optimizador>_semilla<N>.json`.

## Reproducibilidad

Modelo Qwen2.5-7B local (2026-09), caché desactivada al medir. Cada
optimizador se compila tres veces, con semillas 0, 1 y 2 y una muestra propia
de 200 ejemplos de entrenamiento por semilla, y se evalúa sobre la misma
prueba a temperatura cero: exactitud y F1 van como media ± desviación entre
compilaciones. COPRO no admite semilla y varía por el muestreo de su
proponente. GEPA usa el mismo modelo local como `reflection_lm`: la cifra no
mide su rendimiento con un reflector más capaz.

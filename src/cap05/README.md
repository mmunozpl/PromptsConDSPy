# Código del capítulo 5 — optimización de demostraciones

Compara las familias de optimización de demostraciones sobre CSIC 2010 con el
modelo local. Reutiliza el corpus y la firma del capítulo 3. Requiere vLLM en
:8000.

## Dependencias

python>=3.10, dspy==3.2.1, vllm==0.24.0, scikit-learn, numpy, datasets.
Entorno: conda `envDSPy` (RTX 5090). Modelo: Qwen2.5-7B-Instruct local.

## Scripts

- `experimentos.py` — registra las cifras `c5-*` en `../comun/cifras.csv`:
  - `tabla` — cero ejemplos, `LabeledFewShot`, `BootstrapFewShot`, su búsqueda
    aleatoria y un ensemble por mayoría; para cada uno exactitud, F1, coste de
    compilación (Mtok) y de inferencia (tok/pred).
  - `ksweep` — barre el número de demostraciones (0,1,2,4,8), registra la
    curva entera y elige el k de mejor media en validación.
  - Ejecutar: `python experimentos.py`. Grupos: `python experimentos.py ksweep`.

## Reproducibilidad

Modelo Qwen2.5-7B local (2026-09), caché desactivada al medir. Cada familia
se compila tres veces, con semillas 0, 1 y 2 y una muestra propia de 200
ejemplos de entrenamiento por semilla, y se evalúa sobre la misma prueba a
temperatura cero: exactitud y F1 van como media ± desviación entre
compilaciones. `BootstrapFinetune` (ajuste de pesos) queda fuera: exige
backend de entrenamiento aparte del servidor vLLM.

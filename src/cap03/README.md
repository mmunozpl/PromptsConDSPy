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
descarga con caché y lo particiona de forma estratificada.

Antes de partir, el cargador hace tres cosas que fijan el banco:

- **Serializa** cada petición a su línea de solicitud y su cuerpo. En este
  espejo el cuerpo va detrás de la cabecera `Content-Length`, sin línea en
  blanco, y ahí lo busca.
- **Filtra** las 513 filas que no empiezan por una línea de solicitud: son
  cargas útiles sueltas, todas etiquetadas como ataque.
- **Deduplica** por texto serializado: de 42.741 peticiones quedan 18.050
  textos distintos, y ninguna petición de la prueba es idéntica a una de
  entrenamiento. `cargar(deduplicar=False)` conserva los repetidos y solo
  sirve para medir la fuga que la deduplicación evita (capítulo 4). Licencias y
procedencia, en el apéndice de *datasets* de la obra completa.

## Modelo local (vLLM)

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.85
```
En RTX 5090/CUDA 13 el *sampler* JIT de flashinfer no compila: de ahí
`VLLM_USE_FLASHINFER_SAMPLER=0` y `--enforce-eager` (ver apéndice A).

## Scripts

- `datos_csic.py` — descarga, serializa, filtra, deduplica y particiona el
  corpus (estratificado, semilla fija). Ejecutar solo: `python datos_csic.py`
  (imprime un resumen y 15 ejemplos al azar).
- `experimentos.py` — mide todo y registra las cifras. Requiere vLLM en :8000.
  - `python experimentos.py` — todos los grupos. `optimizar` va antes que
    `baselines`, `confusion`, `ofuscacion` y `asr`, que leen el artefacto
    compilado; el orden por omisión mide `baselines` primero, así que tras
    cambiar los datos se lanza con los grupos escritos en el orden de abajo.
  - `python experimentos.py optimizar clasico fraccion baselines confusion
    ofuscacion asr calibracion` — grupos concretos.
  - `python experimentos.py emparejado intervalos`: inyección con
    predicciones emparejadas (las mismas peticiones anómalas sin y con la línea
    hostil, temperatura cero, sin caché, salidas malformadas contadas aparte) y,
    sin LM, el intervalo de Wilson de la exhaustividad y los totales de la tabla
    emparejada. `clasico` registra además la matriz de confusión del TF-IDF y
    `fraccion` cuántas de las peticiones que marca el filtro léxico son
    anómalas.
  - Registra: exactitud/F1/coste de Predict, ChainOfThought y compilado (N=3
    corridas sin caché, media ± desviación); matriz de confusión y κ del
    compilado; robustez ante ofuscación e inyección (ASR); calibración (ECE
    bruto vs reescalado); baseline clásico TF-IDF+LogReg; fracción que llega al
    LM; coste de compilación. El artefacto compilado se guarda en
    `salidas/clasificador_compilado.json`.

## Reproducibilidad

Métricas de LM: media ± desviación sobre N=3 corridas con la caché desactivada.
Modelo: Qwen2.5-7B-Instruct local (2026-07); las cifras son de
ese modelo y ese corpus.

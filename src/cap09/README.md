# Código del capítulo 9 — privacidad

Detección de PII y patrón de delegación con modelo local (vLLM). Requiere vLLM
en :8000. Los frentes que exigen ENTRENAMIENTO (privacidad diferencial DP-SGD,
canarios de memorización) quedan fuera: el banco solo sirve el modelo, no lo
entrena.

## Dependencias

python>=3.10, dspy==3.2.1, vllm==0.24.0, pydantic, numpy, datasets.
Entorno: conda `envDSPy` (RTX 5090). Modelo: Qwen2.5-7B-Instruct local.

## Datos

`ai4privacy/pii-masking-200k` (HuggingFace): textos en inglés con menciones de
PII etiquetadas por tipo (valor, span, categoría). Se mapean a categorías
legibles (persona, contacto, localización, financiero).

## Scripts

- `experimentos.py` — registra las cifras `c9-*` en `../comun/cifras.csv`:
  - `pii` — detector de PII con firma tipada; F1 global (precisión,
    exhaustividad) y exhaustividad por tipo.
  - `latencia` — latencia p50 del patrón de delegación (detectar+redactar+
    responder) frente a responder de un paso.
  - Ejecutar: `python experimentos.py`.

## Reproducibilidad

Modelo Qwen2.5-7B local (2026-07), caché desactivada al medir. Fuera del banco
(exigen entrenamiento o el flujo de delegación completo con jueces de fuga):
exposición de canarios, DP-SGD, demostraciones sintéticas DP, barrido de lambda,
caso PAPILLON de extremo a extremo. Tras ejecutar, `exportar_cifras.py` +
`verificar_cifras.py`.

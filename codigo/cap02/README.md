# Código del capítulo 2

## Dependencias
python>=3.10, `dspy` (3.2.1), `python-dotenv`. Modelo: OpenAI (clave en `.env`)
o local con vLLM (`vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`; ver el
apéndice A del libro).

## Configuración
Copiar `.env.example` a `.env` y rellenar la clave. El `.env` real no se
versiona.

## Scripts
- `primer_programa.py` — configura el LM, declara una firma y predice; imprime
  la predicción y el *prompt* generado (`inspect_history`). No registra cifras.
  Ejecutar: `python primer_programa.py` (requiere `OPENAI_API_KEY` o un vLLM
  local en :8000).

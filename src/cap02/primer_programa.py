# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""primer programa DSPy: configura el LM, declara una firma y predice.

Código ilustrativo del capítulo 2; no registra cifras. Requiere una clave de
API (variable de entorno, p. ej. OPENAI_API_KEY) o un modelo local servido por
vLLM (endpoint compatible con OpenAI en :8000). Ver README.md y el apéndice A.
"""
import os

import dspy
from dotenv import load_dotenv


def configurar_lm() -> dspy.LM:
    """configura el LM por defecto a partir del entorno.

    Usa OpenAI si existe OPENAI_API_KEY; en su defecto, un modelo local
    servido por vLLM (endpoint compatible con OpenAI en :8000). Devuelve el
    LM ya fijado como predeterminado de DSPy.

    Returns:
        El modelo de lenguaje configurado.
    """
    if os.environ.get("OPENAI_API_KEY"):
        lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
        # alternativa por API: dspy.LM("anthropic/claude-haiku-4-5", ...)
    else:
        # modelo local: requiere un vLLM sirviendo el modelo en :8000 (ver
        # apéndice A: 'vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000')
        lm = dspy.LM("hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
                     api_base="http://localhost:8000/v1", api_key="local",
                     temperature=0.0)
    dspy.configure(lm=lm)
    return lm


class Clasificar(dspy.Signature):
    """clasifica un mensaje de log de seguridad por su intención."""

    texto: str = dspy.InputField(desc="línea de log o mensaje crudo")
    etiqueta: str = dspy.OutputField(desc="categoría breve, una palabra")


def main() -> None:
    load_dotenv()          # carga claves desde .env si existe
    configurar_lm()

    # forma rápida: la firma como cadena 'entrada -> salida'
    responder = dspy.Predict("pregunta -> respuesta")
    print(responder(pregunta="¿qué denota una firma en DSPy?").respuesta)

    # forma tipada: la firma como clase, con razonamiento intermedio
    clasificar = dspy.ChainOfThought(Clasificar)
    pred = clasificar(texto="conexión SSH fallida desde 10.0.0.5")
    print("etiqueta:", pred.etiqueta)

    # se inspecciona el prompt que DSPy generó y envió al modelo
    dspy.inspect_history(n=1)


if __name__ == "__main__":
    main()

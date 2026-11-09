# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""prueba de humo del capítulo 2, sin red ni clave de API.

Comprueba con un modelo simulado (dspy.utils.DummyLM) tres conductas que el
capítulo describe en DSPy 3.2.1: la composición de Triaje, la traza delimitada
con dspy.context(trace=[]) y la recuperación del adaptador de chat con el de
JSON cuando el análisis falla. No registra cifras.
"""
from typing import Literal

import dspy
from dspy.utils import DummyLM


class Clasificar(dspy.Signature):
    """clasifica un mensaje de log de seguridad por su intención."""

    texto: str = dspy.InputField(desc="línea de log o mensaje crudo")
    etiqueta: Literal["benigno", "sospechoso", "critico"] = (
        dspy.OutputField())


class Triaje(dspy.Module):
    """clasifica un mensaje y lo resume conforme a su etiqueta."""

    def __init__(self) -> None:
        super().__init__()
        self.clasificar = dspy.ChainOfThought(Clasificar)
        self.resumir = dspy.Predict("texto, etiqueta -> resumen")

    def forward(self, texto: str) -> dspy.Prediction:
        """encadena la clasificación y el resumen.

        Args:
            texto: línea de log o mensaje crudo.

        Returns:
            La predicción con la etiqueta y el resumen.
        """
        c = self.clasificar(texto=texto)
        r = self.resumir(texto=texto, etiqueta=c.etiqueta)
        return dspy.Prediction(etiqueta=c.etiqueta, resumen=r.resumen)


def comprobar(condicion: bool, mensaje: str) -> None:
    """eleva un error si no se cumple la condición.

    Args:
        condicion: resultado de la comprobación.
        mensaje: descripción del fallo.

    Raises:
        RuntimeError: si la condición es falsa.
    """
    if not condicion:
        raise RuntimeError(mensaje)


def probar_traza() -> None:
    """ejecuta Triaje con respuestas fijadas y lee traza e historial."""
    lm = DummyLM([{"reasoning": "fallo de autenticación",
                   "etiqueta": "sospechoso"},
                  {"resumen": "intento SSH fallido"}])
    dspy.configure(lm=lm)
    programa = Triaje()
    with dspy.context(trace=[]):
        resultado = programa(texto="conexion SSH fallida desde 10.0.0.5")
        pasos = list(dspy.settings.trace)
    for predictor, entradas, salida in pasos:
        print(type(predictor).__name__, entradas, dict(salida))
    comprobar(resultado.etiqueta == "sospechoso", "etiqueta inesperada")
    comprobar(len(pasos) == 2, f"la traza tiene {len(pasos)} pasos")
    comprobar(len(lm.history) == 2, "el historial no tiene dos llamadas")
    # los tokens se leen en el historial, no en la traza
    comprobar("usage" in lm.history[-1], "el historial no trae el uso")
    print("traza: 2 pasos; historial: 2 llamadas con su uso")


def probar_recuperacion() -> None:
    """provoca una etiqueta fuera del Literal y cuenta las llamadas."""
    lm = DummyLM([{"reasoning": "x", "etiqueta": "rarisimo"}] * 3)
    dspy.configure(lm=lm)
    try:
        dspy.ChainOfThought(Clasificar)(texto="conexion SSH fallida")
    except dspy.utils.exceptions.AdapterParseError:
        # se espera la excepción tras el segundo intento con JSONAdapter
        comprobar(len(lm.history) == 2,
                  f"{len(lm.history)} llamadas, se esperaban 2")
        print("recuperación: 2 llamadas (chat y JSON) y AdapterParseError")
        return
    raise RuntimeError("la etiqueta fuera del Literal no elevó el error")


def main() -> None:
    probar_traza()
    probar_recuperacion()


if __name__ == "__main__":
    main()

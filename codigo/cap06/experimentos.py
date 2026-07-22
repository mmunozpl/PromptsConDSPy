# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 6: optimización de instrucciones.

Compara, sobre el corpus CSIC 2010 y el modelo local, los optimizadores de
instrucciones —COPRO, MIPROv2 y GEPA— midiendo para cada uno la exactitud, el
F1 macro, el coste de compilación y el de inferencia frente al baseline sin
instrucción optimizada. Cada cifra se registra en cifras.csv.

Uso:  python experimentos.py            # copro, mipro, gepa, baseline
      python experimentos.py copro      # un optimizador concreto
"""
from __future__ import annotations

import sys
from pathlib import Path

import dspy
import numpy as np
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "cap03"))
from comun.determinismo import fijar_semillas  # noqa: E402
from comun.registro import registrar  # noqa: E402
import datos_csic as D  # noqa: E402
from experimentos import (Clasificar, HILOS, MODELO, API_BASE,  # noqa: E402
                          configurar, evaluar, sin_cache, tokens_medios)

# resultados acumulados para la tabla (acc de cada optimizador)
_ACC = {}


def metrica(ej, pred, traza=None):
    return ej.etiqueta == getattr(pred, "etiqueta", None)


def metrica_fb(ej, pred, traza=None, pred_name=None, pred_trace=None):
    """métrica con feedback textual para GEPA."""
    ok = ej.etiqueta == getattr(pred, "etiqueta", None)
    fb = ("correcto" if ok else
          f"incorrecto: era '{ej.etiqueta}', dijo "
          f"'{getattr(pred, 'etiqueta', '?')}'")
    return dspy.Prediction(score=float(ok), feedback=fb)


def _f1(r):
    return f1_score(r["gold"], r["pred"], labels=["benigno", "ataque"],
                    average="macro")


def _medir(nombre, prog, prueba, tok_comp) -> None:
    with sin_cache() as lm:
        lm.history.clear()
        r = evaluar(prog, prueba)
        tok_inf = tokens_medios(lm, len(prueba))
    acc, f1 = r["acc"], _f1(r)
    _ACC[nombre] = acc
    registrar(f"c6-{nombre}-acc", acc * 100, decimales=1)
    registrar(f"c6-{nombre}-f1", f1 * 100, decimales=1)
    registrar(f"c6-{nombre}-comp", tok_comp / 1e6, decimales=2)
    registrar(f"c6-{nombre}-inf", tok_inf, decimales=0)
    print(f"  {nombre:8s}: acc={acc:.3f} f1={f1:.3f} "
          f"comp={tok_comp/1e6:.2f}M inf={tok_inf:.0f}")


def baseline(datos) -> None:
    _, _, prueba = datos
    fijar_semillas(0)
    _medir("base", dspy.ChainOfThought(Clasificar), prueba, 0)


def copro(datos) -> None:
    entreno, desarrollo, prueba = datos
    fijar_semillas(0)
    with sin_cache() as lm:
        lm.history.clear()
        opt = dspy.COPRO(metric=metrica, breadth=3, depth=2)
        prog = opt.compile(
            dspy.ChainOfThought(Clasificar), trainset=desarrollo,
            eval_kwargs={"num_threads": HILOS, "display_progress": False})
        tok = sum(int((h.get("usage") or {}).get("total_tokens", 0))
                  for h in lm.history)
    _medir("copro", prog, prueba, tok)


def mipro(datos) -> None:
    entreno, desarrollo, prueba = datos
    fijar_semillas(0)
    with sin_cache() as lm:
        lm.history.clear()
        opt = dspy.MIPROv2(metric=metrica, auto="light", num_threads=HILOS)
        prog = opt.compile(dspy.ChainOfThought(Clasificar),
                           trainset=entreno[:200], valset=desarrollo,
                           requires_permission_to_run=False)
        tok = sum(int((h.get("usage") or {}).get("total_tokens", 0))
                  for h in lm.history)
    _medir("mipro", prog, prueba, tok)


def gepa(datos) -> None:
    entreno, desarrollo, prueba = datos
    fijar_semillas(0)
    reflexion = dspy.LM(MODELO, api_base=API_BASE, api_key="local",
                        temperature=1.0, max_tokens=2000, cache=False)
    with sin_cache() as lm:
        lm.history.clear()
        opt = dspy.GEPA(metric=metrica_fb, reflection_lm=reflexion,
                        max_metric_calls=600, reflection_minibatch_size=6)
        prog = opt.compile(dspy.ChainOfThought(Clasificar),
                           trainset=entreno[:200], valset=desarrollo)
        tok = sum(int((h.get("usage") or {}).get("total_tokens", 0))
                  for h in lm.history)
    _medir("gepa", prog, prueba, tok)


GRUPOS = {"baseline": baseline, "copro": copro, "mipro": mipro, "gepa": gepa}


def main() -> None:
    configurar()
    datos = D.cargar()
    print(f"datos: {len(datos[0])}/{len(datos[1])}/{len(datos[2])}")
    for nombre in (sys.argv[1:] or ["baseline", "copro", "mipro", "gepa"]):
        print(f"\n### {nombre} ###")
        GRUPOS[nombre](datos)


if __name__ == "__main__":
    main()

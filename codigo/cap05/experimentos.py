# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 5: optimización de demostraciones.

Compara, sobre el corpus CSIC 2010 y el modelo local, las familias de
optimización de demostraciones —cero ejemplos, etiquetadas, arranque
(bootstrap), arranque con búsqueda aleatoria y ensemble— midiendo para cada una
exactitud, F1 macro, coste de compilación y coste de inferencia. Además, barre
el número de demostraciones para hallar el k que rinde mejor. Cada cifra se
registra en cifras.csv.

Uso:  python experimentos.py            # tabla completa + k-óptimo
      python experimentos.py ksweep     # solo el barrido de k
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
from experimentos import (Clasificar, HILOS, configurar, evaluar,  # noqa: E402
                          sin_cache, tokens_medios)


def metrica(ej, pred, traza=None):
    return ej.etiqueta == getattr(pred, "etiqueta", None)


def _f1(r):
    return f1_score(r["gold"], r["pred"], labels=["benigno", "ataque"],
                    average="macro")


def _compilar(nombre, entreno, desarrollo):
    """devuelve (programa_compilado, tokens_de_compilacion)."""
    base = dspy.ChainOfThought(Clasificar)
    with sin_cache() as lm:
        lm.history.clear()
        if nombre == "zeroshot":
            prog = base
        elif nombre == "labeled":
            opt = dspy.LabeledFewShot(k=4)
            prog = opt.compile(base, trainset=entreno)
        elif nombre == "bootstrap":
            opt = dspy.BootstrapFewShot(metric=metrica,
                                        max_bootstrapped_demos=4,
                                        max_labeled_demos=4)
            prog = opt.compile(base, trainset=entreno[:200])
        elif nombre == "random":
            opt = dspy.BootstrapFewShotWithRandomSearch(
                metric=metrica, max_bootstrapped_demos=4, max_labeled_demos=4,
                num_candidate_programs=6, num_threads=HILOS)
            prog = opt.compile(base, trainset=entreno[:200], valset=desarrollo)
        else:
            raise ValueError(nombre)
        tok = sum(int((h.get("usage") or {}).get("total_tokens", 0))
                  for h in lm.history)
    return prog, tok


def _ensemble(entreno, desarrollo):
    """ensemble por mayoría de tres programas de arranque con semillas distintas."""
    from dspy.teleprompt import Ensemble
    progs = []
    tok = 0
    with sin_cache() as lm:
        for semilla in range(3):
            fijar_semillas(semilla)
            lm.history.clear()
            opt = dspy.BootstrapFewShot(metric=metrica,
                                        max_bootstrapped_demos=4,
                                        max_labeled_demos=4)
            progs.append(opt.compile(dspy.ChainOfThought(Clasificar),
                                     trainset=entreno[:200]))
            tok += sum(int((h.get("usage") or {}).get("total_tokens", 0))
                       for h in lm.history)
    ens = Ensemble(reduce_fn=dspy.majority).compile(progs)
    return ens, tok, progs


def tabla(datos) -> None:
    entreno, desarrollo, prueba = datos
    filas = []
    mejor_bootstrap = None
    for nombre in ["zeroshot", "labeled", "bootstrap", "random"]:
        fijar_semillas(0)
        prog, tok_comp = _compilar(nombre, entreno, desarrollo)
        with sin_cache() as lm:
            lm.history.clear()
            r = evaluar(prog, prueba)
            tok_inf = tokens_medios(lm, len(prueba))
        acc, f1 = r["acc"], _f1(r)
        registrar(f"c5-{nombre}-acc", acc * 100, decimales=1)
        registrar(f"c5-{nombre}-f1", f1 * 100, decimales=1)
        registrar(f"c5-{nombre}-comp", tok_comp / 1e6, decimales=2)
        registrar(f"c5-{nombre}-inf", tok_inf, decimales=0)
        filas.append((nombre, acc, f1, tok_comp / 1e6, tok_inf))
        if nombre == "bootstrap":
            mejor_bootstrap = acc
        print(f"  {nombre:10s}: acc={acc:.3f} f1={f1:.3f} "
              f"comp={tok_comp/1e6:.2f}M inf={tok_inf:.0f}")

    # ensemble frente al mejor candidato individual de arranque
    fijar_semillas(0)
    ens, tok_comp_e, progs = _ensemble(entreno, desarrollo)
    with sin_cache() as _:
        acc_ind = max(evaluar(p, prueba)["acc"] for p in progs)
        r_ens = evaluar(ens, prueba)
    acc_e = r_ens["acc"]
    registrar("c5-ensemble-ganancia", (acc_e - acc_ind) * 100, decimales=1)
    print(f"  ensemble  : acc={acc_e:.3f} (mejor individual={acc_ind:.3f}, "
          f"ganancia={(acc_e-acc_ind)*100:.1f}pp)")


def ksweep(datos) -> None:
    """barre el número de demostraciones y elige el k que maximiza validación."""
    entreno, desarrollo, prueba = datos
    mejor_k, mejor_acc = 0, -1.0
    for k in [0, 1, 2, 4, 8]:
        fijar_semillas(0)
        base = dspy.ChainOfThought(Clasificar)
        with sin_cache() as _:
            if k == 0:
                prog = base
            else:
                opt = dspy.BootstrapFewShot(metric=metrica,
                                            max_bootstrapped_demos=k,
                                            max_labeled_demos=k)
                prog = opt.compile(base, trainset=entreno[:200])
            acc = evaluar(prog, desarrollo)["acc"]
        print(f"  k={k}: acc validación={acc:.3f}")
        if acc > mejor_acc:
            mejor_acc, mejor_k = acc, k
    registrar("c5-k-optimo", mejor_k, decimales=0)
    print(f"  k óptimo = {mejor_k} (acc validación {mejor_acc:.3f})")


GRUPOS = {"tabla": tabla, "ksweep": ksweep}


def main() -> None:
    configurar()
    datos = D.cargar()
    print(f"datos: {len(datos[0])}/{len(datos[1])}/{len(datos[2])}")
    for nombre in (sys.argv[1:] or list(GRUPOS)):
        print(f"\n### {nombre} ###")
        GRUPOS[nombre](datos)


if __name__ == "__main__":
    main()

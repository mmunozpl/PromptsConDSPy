# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 5: optimización de demostraciones.

Compara, sobre el corpus CSIC 2010 y el modelo local, las familias de
optimización de demostraciones —cero ejemplos, etiquetadas, arranque
(bootstrap), arranque con búsqueda aleatoria y ensemble— midiendo para cada una
exactitud, F1 macro, coste de compilación y coste de inferencia. Además, barre
el número de demostraciones para hallar el k que rinde mejor.

Cada familia se compila N_CORRIDAS veces, cada vez con su semilla y con su
propia muestra de 200 ejemplos de entrenamiento, y se evalúa sobre la misma
prueba a temperatura cero: la cifra es la media con su desviación entre
compilaciones. Cada cifra se registra en cifras.csv.

Uso:  python experimentos.py            # tabla completa + k-óptimo
      python experimentos.py ksweep     # solo el barrido de k
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import dspy
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "cap03"))
from comun.determinismo import fijar_semillas  # noqa: E402
from comun.registro import registrar  # noqa: E402
import datos_csic as D  # noqa: E402
from experimentos import (Clasificar, HILOS, configurar, evaluar,  # noqa: E402
                          sin_cache, tokens_medios)

N_CORRIDAS = 3
N_ENTRENO = 200
FAMILIAS = ["zeroshot", "labeled", "bootstrap", "random"]


def metrica(ej, pred, traza=None):
    return ej.etiqueta == getattr(pred, "etiqueta", None)


def _f1(r):
    return f1_score(r["gold"], r["pred"], labels=["benigno", "ataque"],
                    average="macro")


def _media_desv(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def muestra_entreno(entreno: list, semilla: int) -> list:
    """muestra de N_ENTRENO ejemplos de entrenamiento propia de la semilla."""
    return random.Random(semilla).sample(entreno, N_ENTRENO)


def _tokens(lm) -> int:
    return sum(int((h.get("usage") or {}).get("total_tokens", 0))
               for h in lm.history)


def _compilar(nombre: str, sub: list, desarrollo: list, k: int = 4):
    """devuelve (programa_compilado, tokens_de_compilacion).

    Args:
        nombre: familia de optimización.
        sub: muestra de entrenamiento de la corrida.
        desarrollo: conjunto de validación, común a todas las corridas.
        k: número de demostraciones.

    Returns:
        El programa compilado y los tokens gastados en compilarlo.
    """
    base = dspy.ChainOfThought(Clasificar)
    with sin_cache() as lm:
        lm.history.clear()
        if nombre == "zeroshot":
            prog = base
        elif nombre == "labeled":
            prog = dspy.LabeledFewShot(k=k).compile(base, trainset=sub)
        elif nombre == "bootstrap":
            opt = dspy.BootstrapFewShot(metric=metrica,
                                        max_bootstrapped_demos=k,
                                        max_labeled_demos=k)
            prog = opt.compile(base, trainset=sub)
        elif nombre == "random":
            opt = dspy.BootstrapFewShotWithRandomSearch(
                metric=metrica, max_bootstrapped_demos=k, max_labeled_demos=k,
                num_candidate_programs=6, num_threads=HILOS)
            prog = opt.compile(base, trainset=sub, valset=desarrollo)
        else:
            raise ValueError(nombre)
        tok = _tokens(lm)
    return prog, tok


def _medir(prog, particion) -> tuple[float, float, float]:
    """exactitud, F1 macro y tokens por predicción, sin caché."""
    with sin_cache() as lm:
        lm.history.clear()
        r = evaluar(prog, particion)
        tok_inf = tokens_medios(lm, len(particion))
    return r["acc"], _f1(r), tok_inf


def tabla(datos) -> None:
    entreno, desarrollo, prueba = datos
    arranques = []
    for nombre in FAMILIAS:
        accs, f1s, comps, infs = [], [], [], []
        for s in tqdm(range(N_CORRIDAS), desc=nombre):
            fijar_semillas(s)
            prog, tok_comp = _compilar(nombre, muestra_entreno(entreno, s),
                                       desarrollo)
            acc, f1, tok_inf = _medir(prog, prueba)
            accs.append(acc); f1s.append(f1)
            comps.append(tok_comp / 1e6); infs.append(tok_inf)
            if nombre == "bootstrap":
                arranques.append((prog, acc))
            print(f"  {nombre:10s} semilla {s}: acc={acc:.3f} f1={f1:.3f} "
                  f"comp={tok_comp/1e6:.2f}M inf={tok_inf:.0f}")
        acc_m, acc_d = _media_desv(accs)
        f1_m, f1_d = _media_desv(f1s)
        registrar(f"c5-{nombre}-acc", acc_m * 100, decimales=1,
                  desv=acc_d * 100)
        registrar(f"c5-{nombre}-f1", f1_m * 100, decimales=1,
                  desv=f1_d * 100)
        registrar(f"c5-{nombre}-comp", float(np.mean(comps)), decimales=2)
        registrar(f"c5-{nombre}-inf", float(np.mean(infs)), decimales=0)
        print(f"  {nombre:10s}: acc={acc_m*100:.1f}±{acc_d*100:.1f} "
              f"f1={f1_m*100:.1f}±{f1_d*100:.1f}")

    # ensemble por mayoría de los tres programas de arranque de la tabla,
    # frente al mejor de ellos
    from dspy.teleprompt import Ensemble
    ens = Ensemble(reduce_fn=dspy.majority).compile(
        [p for p, _ in arranques])
    acc_e, _, _ = _medir(ens, prueba)
    acc_ind = max(a for _, a in arranques)
    registrar("c5-ensemble-ganancia", (acc_e - acc_ind) * 100, decimales=1)
    print(f"  ensemble  : acc={acc_e:.3f} (mejor individual={acc_ind:.3f}, "
          f"ganancia={(acc_e-acc_ind)*100:.1f}pp)")


def ksweep(datos) -> None:
    """barre el número de demostraciones; elige el k de mejor media."""
    entreno, desarrollo, _ = datos
    mejor_k, mejor_acc = 0, -1.0
    for k in [0, 1, 2, 4, 8]:
        accs = []
        for s in tqdm(range(N_CORRIDAS), desc=f"k={k}"):
            fijar_semillas(s)
            nombre = "zeroshot" if k == 0 else "bootstrap"
            prog, _ = _compilar(nombre, muestra_entreno(entreno, s),
                                desarrollo, k=max(k, 1))
            accs.append(_medir(prog, desarrollo)[0])
        m, d = _media_desv(accs)
        # la curva entera va al registro: el óptimo solo no dice si es plana
        registrar(f"c5-k{k}-acc", m * 100, decimales=1, desv=d * 100)
        print(f"  k={k}: acc validación={m*100:.1f}±{d*100:.1f}")
        if m > mejor_acc:
            mejor_acc, mejor_k = m, k
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

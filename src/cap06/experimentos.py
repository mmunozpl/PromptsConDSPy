# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 6: optimización de instrucciones.

Compara, sobre el corpus CSIC 2010 y el modelo local, los optimizadores de
instrucciones —COPRO, MIPROv2 y GEPA— midiendo para cada uno la exactitud, el
F1 macro, el coste de compilación y el de inferencia frente al baseline sin
instrucción optimizada.

Cada optimizador se compila N_CORRIDAS veces, cada vez con su semilla y con su
propia muestra de 200 ejemplos de entrenamiento, y se evalúa sobre la misma
prueba a temperatura cero: la cifra es la media con su desviación entre
compilaciones. Cada cifra se registra en cifras.csv.

Uso:  python experimentos.py            # baseline, copro, mipro, gepa
      python experimentos.py copro      # un optimizador concreto
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
from experimentos import (Clasificar, HILOS, MODELO, API_BASE,  # noqa: E402
                          configurar, evaluar, sin_cache, tokens_medios)

N_CORRIDAS = 3
N_ENTRENO = 200


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


def _media_desv(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def muestra_entreno(entreno: list, semilla: int) -> list:
    """muestra de N_ENTRENO ejemplos de entrenamiento propia de la semilla."""
    return random.Random(semilla).sample(entreno, N_ENTRENO)


def _tokens(lm) -> int:
    return sum(int((h.get("usage") or {}).get("total_tokens", 0))
               for h in lm.history)


def _compilar(nombre: str, sub: list, desarrollo: list, semilla: int):
    """devuelve (programa_compilado, tokens_de_compilacion)."""
    base = dspy.ChainOfThought(Clasificar)
    with sin_cache() as lm:
        lm.history.clear()
        if nombre == "base":
            prog = base
        elif nombre == "copro":
            # COPRO no admite semilla: varía por el muestreo del proponente
            opt = dspy.COPRO(metric=metrica, breadth=3, depth=2)
            prog = opt.compile(
                base, trainset=desarrollo,
                eval_kwargs={"num_threads": HILOS,
                             "display_progress": False})
        elif nombre == "mipro":
            opt = dspy.MIPROv2(metric=metrica, auto="light",
                               num_threads=HILOS, seed=semilla)
            prog = opt.compile(base, trainset=sub, valset=desarrollo,
                               requires_permission_to_run=False)
        elif nombre == "gepa":
            reflexion = dspy.LM(MODELO, api_base=API_BASE, api_key="local",
                                temperature=1.0, max_tokens=2000,
                                cache=False)
            opt = dspy.GEPA(metric=metrica_fb, reflection_lm=reflexion,
                            max_metric_calls=600,
                            reflection_minibatch_size=6, seed=semilla)
            prog = opt.compile(base, trainset=sub, valset=desarrollo)
        else:
            raise ValueError(nombre)
        tok = _tokens(lm)
    return prog, tok


def _medir(prog, prueba) -> tuple[float, float, float]:
    """exactitud, F1 macro y tokens por predicción, sin caché."""
    with sin_cache() as lm:
        lm.history.clear()
        r = evaluar(prog, prueba)
        tok_inf = tokens_medios(lm, len(prueba))
    return r["acc"], _f1(r), tok_inf


def _optimizador(nombre: str, datos) -> None:
    entreno, desarrollo, prueba = datos
    accs, f1s, comps, infs = [], [], [], []
    for s in tqdm(range(N_CORRIDAS), desc=nombre):
        fijar_semillas(s)
        prog, tok_comp = _compilar(nombre, muestra_entreno(entreno, s),
                                   desarrollo, s)
        acc, f1, tok_inf = _medir(prog, prueba)
        accs.append(acc); f1s.append(f1)
        comps.append(tok_comp / 1e6); infs.append(tok_inf)
        print(f"  {nombre:8s} semilla {s}: acc={acc:.3f} f1={f1:.3f} "
              f"comp={tok_comp/1e6:.2f}M inf={tok_inf:.0f}")
        if nombre != "base":
            ruta = Path(__file__).with_name("salidas")
            ruta.mkdir(exist_ok=True)
            prog.save(str(ruta / f"{nombre}_semilla{s}.json"))
    acc_m, acc_d = _media_desv(accs)
    f1_m, f1_d = _media_desv(f1s)
    registrar(f"c6-{nombre}-acc", acc_m * 100, decimales=1, desv=acc_d * 100)
    registrar(f"c6-{nombre}-f1", f1_m * 100, decimales=1, desv=f1_d * 100)
    registrar(f"c6-{nombre}-comp", float(np.mean(comps)), decimales=2)
    registrar(f"c6-{nombre}-inf", float(np.mean(infs)), decimales=0)
    print(f"  {nombre:8s}: acc={acc_m*100:.1f}±{acc_d*100:.1f} "
          f"f1={f1_m*100:.1f}±{f1_d*100:.1f}")


def baseline(datos) -> None:
    _optimizador("base", datos)


def copro(datos) -> None:
    _optimizador("copro", datos)


def mipro(datos) -> None:
    _optimizador("mipro", datos)


def gepa(datos) -> None:
    _optimizador("gepa", datos)


GRUPOS = {"baseline": baseline, "copro": copro, "mipro": mipro, "gepa": gepa}


def main() -> None:
    configurar()
    datos = D.cargar()
    print(f"datos: {len(datos[0])}/{len(datos[1])}/{len(datos[2])}")
    for nombre in (sys.argv[1:] or list(GRUPOS)):
        print(f"\n### {nombre} ###")
        GRUPOS[nombre](datos)


if __name__ == "__main__":
    main()

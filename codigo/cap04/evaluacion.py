# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 4: evaluación rigurosa del clasificador del SOC.

Reutiliza el corpus CSIC 2010 y el clasificador compilado del capítulo 3, y
mide las cifras de cabecera de la evaluación: la tabla con exactitud, F1, kappa
y MCC; el desglose de varianza (muestreo frente a decodificación); la
consistencia por ejemplo; las rebanadas por clase; la fuga por vecindad; los
percentiles de latencia; el umbral óptimo por costes; el gap de contaminación;
y el perfil de fallos. Cada cifra se registra en cifras.csv.

Uso:  python experimentos.py           # todo (requiere vLLM en :8000)
      python experimentos.py tabla varianza  # grupos concretos
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (cohen_kappa_score, f1_score,
                             matthews_corrcoef)
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "cap03"))
from comun.determinismo import fijar_semillas  # noqa: E402
from comun.registro import registrar  # noqa: E402
import datos_csic as D  # noqa: E402
from experimentos import (Clasificar, ClasificarConfianza, HILOS, MODELO,  # noqa: E402
                          API_BASE, configurar, evaluar, lm_medida, sin_cache,
                          cargar_compilado)

N_CORRIDAS = 3
K_VOTOS = 5


def _pred_paralelo(prog, ejemplos):
    with ThreadPoolExecutor(max_workers=HILOS) as pool:
        return list(pool.map(lambda e: prog(peticion=e.peticion), ejemplos))


# --- grupo 1: la tabla exactitud / F1 / kappa / MCC ------------------------

def medir_tabla(datos) -> None:
    _, _, prueba = datos
    prog = cargar_compilado()
    with sin_cache() as _:
        r = evaluar(prog, prueba)
    gold, pred = r["gold"], r["pred"]
    acc = float(np.mean([g == p for g, p in zip(gold, pred)]))
    f1 = f1_score(gold, pred, labels=["benigno", "ataque"], average="macro")
    # a binario numérico para kappa/MCC
    y = [1 if g == "ataque" else 0 for g in gold]
    yh = [1 if p == "ataque" else 0 for p in pred]
    kappa = cohen_kappa_score(y, yh)
    mcc = matthews_corrcoef(y, yh)
    registrar("c4-acc", acc * 100, decimales=1)
    registrar("c4-f1", f1 * 100, decimales=1)
    registrar("c4-kappa", kappa, decimales=3)
    registrar("c4-mcc", mcc, decimales=3)
    print(f"  tabla: acc={acc:.3f} f1={f1:.3f} kappa={kappa:.3f} mcc={mcc:.3f}")


# --- grupo 2: acc como media +- desviación (N corridas) --------------------

def medir_acc_media(datos) -> None:
    _, _, prueba = datos
    prog = cargar_compilado()
    accs = []
    with sin_cache(temperatura=0.7) as _:
        for c in range(N_CORRIDAS):
            fijar_semillas(c)
            r = evaluar(prog, prueba)
            accs.append(r["acc"])
    m, d = float(np.mean(accs)), float(np.std(accs))
    registrar("c4-acc-media", m * 100, decimales=1, desv=d * 100)
    print(f"  acc media: {m*100:.1f} ± {d*100:.1f} sobre {N_CORRIDAS} corridas")


# --- grupo 3: desglose de varianza (muestreo vs decodificación) ------------

def medir_varianza(datos) -> None:
    prog = cargar_compilado()
    # decodificación: misma partición, N corridas a temperatura > 0
    _, _, prueba = datos
    accs_dec = []
    with sin_cache(temperatura=0.7) as _:
        for c in range(N_CORRIDAS):
            fijar_semillas(c)
            accs_dec.append(evaluar(prog, prueba)["acc"])
    var_dec = float(np.std(accs_dec)) * 100
    # muestreo: distintas particiones (semilla de datos), temperatura 0
    accs_mue = []
    with sin_cache(temperatura=0.0) as _:
        for s in range(N_CORRIDAS):
            _, _, pr = D.cargar(semilla=s)
            accs_mue.append(evaluar(prog, pr)["acc"])
    var_mue = float(np.std(accs_mue)) * 100
    registrar("c4-var-decodificacion", var_dec, decimales=1)
    registrar("c4-var-muestreo", var_mue, decimales=1)
    print(f"  varianza: decodificación={var_dec:.1f}pp  muestreo={var_mue:.1f}pp")


# --- grupo 4: consistencia por ejemplo (fracción vacilante) ----------------

def medir_consistencia(datos) -> None:
    _, _, prueba = datos
    prog = cargar_compilado()
    votos = {i: [] for i in range(len(prueba))}
    with sin_cache(temperatura=0.7) as _:
        for _k in range(K_VOTOS):
            preds = _pred_paralelo(prog, prueba)
            for i, p in enumerate(preds):
                votos[i].append(getattr(p, "etiqueta", "benigno"))
    vacilantes = sum(1 for i in votos if len(set(votos[i])) > 1)
    frac = vacilantes / len(prueba) * 100
    registrar("c4-fraccion-vacilante", frac, decimales=1)
    print(f"  vacilantes: {vacilantes}/{len(prueba)} = {frac:.1f}% "
          f"(desacuerdo en {K_VOTOS} muestras)")


# --- grupo 5: rebanadas (por clase y por longitud) -------------------------

def medir_rebanadas(datos) -> None:
    _, _, prueba = datos
    prog = cargar_compilado()
    with sin_cache() as _:
        r = evaluar(prog, prueba)
    gold, pred = r["gold"], r["pred"]
    # F1 por clase
    f1_ben = f1_score(gold, pred, labels=["benigno", "ataque"],
                      average=None)[0]
    f1_ata = f1_score(gold, pred, labels=["benigno", "ataque"],
                      average=None)[1]
    registrar("c4-f1-benigno", f1_ben * 100, decimales=1)
    registrar("c4-f1-ataque", f1_ata * 100, decimales=1)
    # rebanada por longitud de la petición (mediana como corte)
    long_ = [len(e.peticion) for e in prueba]
    corte = float(np.median(long_))
    cortos = [(g, p) for g, p, L in zip(gold, pred, long_) if L <= corte]
    largos = [(g, p) for g, p, L in zip(gold, pred, long_) if L > corte]
    acc_cortos = np.mean([g == p for g, p in cortos]) * 100
    acc_largos = np.mean([g == p for g, p in largos]) * 100
    registrar("c4-acc-cortos", acc_cortos, decimales=1)
    registrar("c4-acc-largos", acc_largos, decimales=1)
    print(f"  rebanadas: F1 benigno={f1_ben:.3f} ataque={f1_ata:.3f}; "
          f"acc cortos={acc_cortos:.1f} largos={acc_largos:.1f}")


# --- grupo 6: fuga por vecindad (train-test) -------------------------------

def medir_vecino_fuga(datos) -> None:
    entreno, _, prueba = datos
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    Atr = vec.fit_transform([e.peticion for e in entreno])
    Ate = vec.transform([e.peticion for e in prueba])
    sims = cosine_similarity(Ate, Atr)
    maxsim = sims.max(axis=1)
    sim_media = float(maxsim.mean())
    casi_gemelos = float((maxsim > 0.9).mean()) * 100
    registrar("c4-vecino-sim", sim_media, decimales=3)
    registrar("c4-vecino-duplicados", casi_gemelos, decimales=1)
    print(f"  vecino: similitud máx media={sim_media:.3f}; "
          f"{casi_gemelos:.1f}% con vecino >0,9")


# --- grupo 7: percentiles de latencia --------------------------------------

def medir_latencia(datos) -> None:
    _, _, prueba = datos
    prog = cargar_compilado()
    lat = []
    with sin_cache() as _:
        for e in prueba[:60]:
            t0 = time.perf_counter()
            try:
                prog(peticion=e.peticion)
            except Exception:
                pass
            lat.append(time.perf_counter() - t0)
    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))
    registrar("c4-latencia-p50", p50, decimales=2)
    registrar("c4-latencia-p95", p95, decimales=2)
    print(f"  latencia: p50={p50:.2f}s  p95={p95:.2f}s")


# --- grupo 8: umbral óptimo por costes -------------------------------------

def _puntuaciones(datos):
    """P(ataque) por ejemplo, de la confianza verbalizada."""
    _, _, prueba = datos
    prog = dspy.Predict(ClasificarConfianza)
    with sin_cache() as _:
        preds = _pred_paralelo(prog, prueba)
    y, score = [], []
    for e, p in zip(prueba, preds):
        et = getattr(p, "etiqueta", "benigno")
        c = getattr(p, "confianza", 0.5)
        try:
            c = float(c)
        except (TypeError, ValueError):
            c = 0.5
        c = min(max(c, 0.0), 1.0)
        score.append(c if et == "ataque" else 1 - c)
        y.append(1 if e.etiqueta == "ataque" else 0)
    return np.array(y), np.array(score)


def medir_umbral_coste(datos) -> None:
    y, score = _puntuaciones(datos)
    c_fp, c_fn = 1.0, 10.0        # falso negativo (ataque perdido) 10x más caro
    umbrales = np.linspace(0.05, 0.95, 19)
    costes = []
    for u in umbrales:
        pred = (score >= u).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        costes.append(c_fp * fp + c_fn * fn)
    i = int(np.argmin(costes))
    registrar("c4-umbral-optimo", float(umbrales[i]), decimales=2)
    print(f"  umbral óptimo={umbrales[i]:.2f} coste={costes[i]:.0f} "
          f"(c_fn={c_fn:.0f}, c_fp={c_fp:.0f})")


# --- grupo 9: gap de contaminación (limpio vs perturbado) ------------------

def medir_contaminacion(datos) -> None:
    _, _, prueba = datos
    prog = cargar_compilado()
    with sin_cache() as _:
        limpio = evaluar(prog, prueba)
        pert = [dspy.Example(peticion=D.ofuscar(e.peticion),
                             etiqueta=e.etiqueta).with_inputs("peticion")
                for e in prueba]
        perturbado = evaluar(prog, pert)
    gap = (limpio["acc"] - perturbado["acc"]) * 100
    registrar("c4-contaminacion-gap", gap, decimales=1)
    print(f"  contaminación: acc limpio={limpio['acc']:.3f} "
          f"perturbado={perturbado['acc']:.3f} gap={gap:.1f}pp")


# --- grupo 10: perfil de fallos --------------------------------------------

def medir_perfil_fallos(datos) -> None:
    _, _, prueba = datos
    prog = cargar_compilado()
    with sin_cache() as _:
        r = evaluar(prog, prueba)
    errores = [(g, p) for g, p in zip(r["gold"], r["pred"]) if g != p]
    total = len(errores) or 1
    fn = sum(1 for g, p in errores if g == "ataque" and p == "benigno")
    fp = sum(1 for g, p in errores if g == "benigno" and p == "ataque")
    registrar("c4-perfil-fn", fn / total * 100, decimales=1)
    registrar("c4-perfil-fp", fp / total * 100, decimales=1)
    print(f"  perfil de fallos: {len(errores)} errores; "
          f"falsos negativos {fn/total*100:.1f}%, falsos positivos "
          f"{fp/total*100:.1f}%")


# --- grupo 11: juez LM frente a la referencia (kappa) ----------------------

class JuezClasificacion(dspy.Signature):
    """juzga si una peticion HTTP es benigna o un ataque (juez independiente)."""

    peticion: str = dspy.InputField()
    veredicto: str = dspy.OutputField(desc="'benigno' o 'ataque'")


def medir_juez_kappa(datos) -> None:
    _, _, prueba = datos
    juez = dspy.Predict(JuezClasificacion)
    with sin_cache() as _:
        preds = _pred_paralelo(juez, prueba)
    y = [1 if e.etiqueta == "ataque" else 0 for e in prueba]
    yj = [1 if getattr(p, "veredicto", "benigno").strip().lower()
          .startswith("ataque") else 0 for p in preds]
    kappa = cohen_kappa_score(y, yj)
    registrar("c4-juez-kappa", kappa, decimales=3)
    print(f"  juez vs referencia: kappa={kappa:.3f}")


GRUPOS = {
    "tabla": medir_tabla,
    "acc-media": medir_acc_media,
    "varianza": medir_varianza,
    "consistencia": medir_consistencia,
    "rebanadas": medir_rebanadas,
    "vecino": medir_vecino_fuga,
    "latencia": medir_latencia,
    "umbral": medir_umbral_coste,
    "contaminacion": medir_contaminacion,
    "perfil": medir_perfil_fallos,
    "juez": medir_juez_kappa,
}


def main() -> None:
    configurar()
    datos = D.cargar()
    print(f"datos: {len(datos[0])}/{len(datos[1])}/{len(datos[2])}")
    for nombre in (sys.argv[1:] or list(GRUPOS)):
        print(f"\n### {nombre} ###")
        GRUPOS[nombre](datos)


if __name__ == "__main__":
    main()

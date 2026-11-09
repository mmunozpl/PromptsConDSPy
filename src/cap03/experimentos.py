# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 3: clasificación binaria de peticiones HTTP.

Mide, sobre el corpus CSIC 2010 y un modelo local servido por vLLM, todas las
cifras de cabecera del capítulo: exactitud y F1 de Predict frente a
ChainOfThought, matriz de confusión, coste en tokens, robustez ante ofuscación
e inyección, calibración, y la mejora por optimización de demostraciones. Cada
cifra se registra en cifras.csv (nunca se teclea a mano).

Uso:  python experimentos.py            # todo (requiere vLLM en :8000)
      python experimentos.py baselines  # solo un grupo
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import dspy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun.determinismo import fijar_semillas  # noqa: E402
from comun.registro import registrar  # noqa: E402
from datos_csic import (cargar, inyectar, ofuscar)  # noqa: E402

MODELO = "hosted_vllm/Qwen/Qwen2.5-7B-Instruct"
API_BASE = "http://localhost:8000/v1"
N_CORRIDAS = 3
HILOS = 32


def configurar(temperatura: float = 0.0) -> None:
    lm = dspy.LM(MODELO, api_base=API_BASE, api_key="local",
                 temperature=temperatura, max_tokens=1500)
    dspy.configure(lm=lm, track_usage=True)


def lm_medida(temperatura: float = 0.0) -> dspy.LM:
    """LM con la caché DESACTIVADA, para medir tokens y latencia reales.

    La caché de DSPy sirve la misma respuesta sin llamar al modelo, lo que
    falsea el conteo de tokens y el reloj. Para las medidas de
    coste y de variabilidad se corre siempre sin caché.
    """
    return dspy.LM(MODELO, api_base=API_BASE, api_key="local",
                   temperature=temperatura, max_tokens=1500, cache=False)


@contextmanager
def sin_cache(temperatura: float = 0.0):
    """fija un LM sin caché en la config GLOBAL (que sí ven los hilos).

    dspy.context no propaga a un ThreadPoolExecutor externo, de modo que las
    medidas de coste deben reconfigurar el LM por defecto. Al salir se restaura
    el LM con caché para no penalizar la optimización.
    """
    lm = lm_medida(temperatura)
    dspy.configure(lm=lm, track_usage=True)
    try:
        yield lm
    finally:
        configurar()


def tokens_medios(lm, n: int) -> float:
    """tokens totales por predicción, leídos del historial del LM."""
    tot = sum(int((h.get("usage") or {}).get("total_tokens", 0))
              for h in lm.history)
    return tot / n if n else 0.0


def latencia_p95(prog, ejemplos, n: int = 40) -> float:
    """p95 de latencia por petición en una pasada SECUENCIAL (sin contención)."""
    lat = []
    for e in ejemplos[:n]:
        t0 = time.perf_counter()
        try:
            prog(peticion=e.peticion)
        except Exception:
            pass
        lat.append(time.perf_counter() - t0)
    return float(np.percentile(lat, 95))


def cargar_compilado():
    """recupera el clasificador compilado del artefacto guardado."""
    ruta = Path(__file__).with_name("salidas") / "clasificador_compilado.json"
    prog = dspy.ChainOfThought(Clasificar)
    prog.load(str(ruta))
    return prog


# --- firmas ----------------------------------------------------------------

class Clasificar(dspy.Signature):
    """clasifica una peticion HTTP como trafico benigno o como ataque web."""

    peticion: str = dspy.InputField(desc="peticion HTTP serializada")
    etiqueta: Literal["benigno", "ataque"] = dspy.OutputField()


class ClasificarConfianza(dspy.Signature):
    """clasifica una peticion HTTP y declara tu confianza en la respuesta."""

    peticion: str = dspy.InputField(desc="peticion HTTP serializada")
    etiqueta: Literal["benigno", "ataque"] = dspy.OutputField()
    confianza: float = dspy.OutputField(desc="confianza entre 0 y 1")


# --- evaluación de un programa sobre una partición -------------------------

def _predecir_uno(programa, ej):
    """devuelve (gold, pred, confianza|None)."""
    try:
        salida = programa(peticion=ej.peticion)
        pred = salida.etiqueta
        conf = float(getattr(salida, "confianza", float("nan"))) \
            if hasattr(salida, "confianza") else None
    except Exception:
        pred, conf = "benigno", None       # una salida malformada cuenta como fallo
    return ej.etiqueta, pred, conf


def evaluar(programa, prueba) -> dict:
    """corre el programa sobre la prueba en paralelo y agrega métricas."""
    with ThreadPoolExecutor(max_workers=HILOS) as pool:
        filas = list(pool.map(lambda e: _predecir_uno(programa, e), prueba))
    gold = [f[0] for f in filas]
    pred = [f[1] for f in filas]
    confs = [f[2] for f in filas]
    acc = float(np.mean([g == p for g, p in zip(gold, pred)]))
    f1 = f1_score(gold, pred, labels=["benigno", "ataque"], average="macro")
    return {"gold": gold, "pred": pred, "acc": acc, "f1": f1, "confs": confs}


def _media_desv(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


# --- grupo 1: baselines Predict vs ChainOfThought --------------------------

def medir_baselines(datos) -> dict:
    """Predict y CoT sobre la prueba, N corridas SIN caché; tabla completa."""
    _, _, prueba = datos
    # las tres configuraciones de la tabla; el compilado se carga del artefacto
    configs = [("predict", lambda: dspy.Predict(Clasificar)),
               ("cot", lambda: dspy.ChainOfThought(Clasificar))]
    ruta = Path(__file__).with_name("salidas") / "clasificador_compilado.json"
    if ruta.exists():
        configs.append(("opt", cargar_compilado))

    resultados = {}
    with sin_cache(temperatura=0.7) as lm:   # temp>0 para exponer variabilidad
        for nombre, ctor in configs:
            accs, f1s, toks = [], [], []
            for corrida in range(N_CORRIDAS):
                fijar_semillas(corrida)
                prog = ctor()
                lm.history.clear()
                r = evaluar(prog, prueba)
                accs.append(r["acc"]); f1s.append(r["f1"])
                toks.append(tokens_medios(lm, len(prueba)))
                print(f"  {nombre} corrida {corrida}: acc={r['acc']:.3f} "
                      f"f1={r['f1']:.3f} tok={toks[-1]:.0f}")
            lat = latencia_p95(ctor(), prueba)   # pasada secuencial limpia
            acc_m, acc_d = _media_desv(accs)
            f1_m, f1_d = _media_desv(f1s)
            tok_m, _ = _media_desv(toks)
            resultados[nombre] = {"acc": (acc_m, acc_d), "f1": (f1_m, f1_d),
                                  "tokens": tok_m, "lat": lat}
            pre = {"predict": "p", "cot": "c", "opt": "o"}[nombre]
            registrar(f"c3-{pre}-acc", acc_m * 100, decimales=1, desv=acc_d * 100)
            registrar(f"c3-{pre}-f1", f1_m * 100, decimales=1, desv=f1_d * 100)
            registrar(f"c3-{pre}-coste", tok_m, decimales=0)
    # las cifras por configuracion salen SOLO de la familia c3-{p,c,o}-*;
    # duplicarlas bajo otro nombre publicaria dos valores para la misma medida
    registrar("c3-coste-escala", resultados["cot"]["tokens"] * 1000 / 1e6,
              decimales=2)  # millones de tokens por cada mil clasificaciones
    registrar("c3-latencia-p95", resultados["cot"]["lat"], decimales=2)
    return resultados


def medir_confusion(datos, prog=None) -> None:
    """matriz de confusión, exactitud por clase y kappa del clasificador dado.

    Se mide sobre el clasificador COMPILADO (el que se desplegaría), no sobre
    el baseline sesgado, para que las cifras describan el sistema real.
    """
    _, _, prueba = datos
    if prog is None:
        prog = cargar_compilado()
    with dspy.context(lm=lm_medida(), track_usage=True):
        r = evaluar(prog, prueba)
    etiquetas = ["benigno", "ataque"]
    mc = confusion_matrix(r["gold"], r["pred"], labels=etiquetas)
    vn, fp = int(mc[0, 0]), int(mc[0, 1])   # benigno = negativo
    fn, vp = int(mc[1, 0]), int(mc[1, 1])   # ataque = positivo
    registrar("c3-vn", vn, decimales=0); registrar("c3-fp", fp, decimales=0)
    registrar("c3-fn", fn, decimales=0); registrar("c3-vp", vp, decimales=0)
    acc_benigno = vn / (vn + fp) if (vn + fp) else 0.0
    acc_ataque = vp / (vp + fn) if (vp + fn) else 0.0
    registrar("c3-acc-benigno", acc_benigno * 100, decimales=1)
    registrar("c3-acc-ataque", acc_ataque * 100, decimales=1)
    kappa = cohen_kappa_score(r["gold"], r["pred"])
    registrar("c3-kappa-modelo", kappa, decimales=3)
    print(f"  confusión compilado: vn={vn} fp={fp} fn={fn} vp={vp} "
          f"kappa={kappa:.3f}  acc/clase benigno={acc_benigno:.3f} "
          f"ataque={acc_ataque:.3f}")


# --- grupo 2: baseline clásico TF-IDF + regresión logística ----------------

def medir_baseline_clasico(datos) -> None:
    entreno, _, prueba = datos
    Xtr = [e.peticion for e in entreno]; ytr = [e.etiqueta for e in entreno]
    Xte = [e.peticion for e in prueba]; yte = [e.etiqueta for e in prueba]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    Atr = vec.fit_transform(Xtr); Ate = vec.transform(Xte)
    clf = LogisticRegression(max_iter=1000, C=5.0)
    clf.fit(Atr, ytr)
    pred = clf.predict(Ate)
    acc = float(np.mean([p == y for p, y in zip(pred, yte)]))
    f1 = f1_score(yte, pred, average="macro")
    registrar("c3-clasico-acc", acc * 100, decimales=1)
    registrar("c3-clasico-f1", f1 * 100, decimales=1)
    # la matriz del baseline, sobre la misma prueba que la del compilado
    mc = confusion_matrix(yte, pred, labels=["benigno", "ataque"])
    for clave, valor in (("vn", mc[0, 0]), ("fp", mc[0, 1]),
                         ("fn", mc[1, 0]), ("vp", mc[1, 1])):
        registrar(f"c3-clasico-{clave}", int(valor), decimales=0)
    print(f"  clásico TF-IDF+LogReg: acc={acc:.3f} f1={f1:.3f} "
          f"vn={mc[0, 0]} fp={mc[0, 1]} fn={mc[1, 0]} vp={mc[1, 1]}")


# --- grupo 3: robustez ante ofuscación -------------------------------------

def medir_ofuscacion(datos, prog=None) -> None:
    _, _, prueba = datos
    if prog is None:
        prog = cargar_compilado()
    # solo los ataques: medir cuántos se siguen detectando tras ofuscar
    ataques = [e for e in prueba if e.etiqueta == "ataque"]
    with dspy.context(lm=lm_medida(), track_usage=True):
        limpio = evaluar(prog, ataques)
        ofus = [dspy.Example(peticion=ofuscar(e.peticion),
                             etiqueta="ataque").with_inputs("peticion")
                for e in ataques]
        ofuscado = evaluar(prog, ofus)
    # exhaustividad = fracción de ataques aún etiquetados 'ataque'
    rec_limpio = np.mean([p == "ataque" for p in limpio["pred"]])
    rec_ofus = np.mean([p == "ataque" for p in ofuscado["pred"]])
    caida = (rec_limpio - rec_ofus) * 100
    registrar("c3-rec-limpio", rec_limpio * 100, decimales=1)
    registrar("c3-rec-ofuscado", rec_ofus * 100, decimales=1)
    registrar("c3-ofuscacion-caida", caida, decimales=1)
    print(f"  ofuscación: recall limpio={rec_limpio:.3f} "
          f"ofuscado={rec_ofus:.3f} caída={caida:.1f}pp")


# --- grupo 4: tasa de éxito de inyección (ASR) -----------------------------

def medir_asr(datos, prog=None) -> None:
    _, _, prueba = datos
    if prog is None:
        prog = cargar_compilado()
    ataques = [e for e in prueba if e.etiqueta == "ataque"]
    iny = [dspy.Example(peticion=inyectar(e.peticion),
                        etiqueta="ataque").with_inputs("peticion")
           for e in ataques]
    with dspy.context(lm=lm_medida(), track_usage=True):
        r = evaluar(prog, iny)
    # éxito del ataque = un ataque real que la inyección logra marcar 'benigno'
    asr = np.mean([p == "benigno" for p in r["pred"]]) * 100
    registrar("c3-asr", asr, decimales=1)
    print(f"  ASR (inyección): {asr:.1f}% de ataques desviados a 'benigno'")


def _predecir_estricto(programa, ej) -> str:
    """devuelve la etiqueta, o 'malformada' si el adaptador no la extrae."""
    try:
        return programa(peticion=ej.peticion).etiqueta
    except Exception:
        return "malformada"


def medir_emparejado(datos, prog=None) -> None:
    """inyección emparejada: las mismas peticiones anómalas, sin y con la línea.

    Separa los fallos previos de los que causa la inyección: cada petición se
    clasifica dos veces en la misma sesión (temperatura cero, sin caché), y
    las salidas malformadas se cuentan aparte en vez de contarse como benigno.
    """
    _, _, prueba = datos
    if prog is None:
        prog = cargar_compilado()
    ataques = [e for e in prueba if e.etiqueta == "ataque"]
    iny = [dspy.Example(peticion=inyectar(e.peticion),
                        etiqueta="ataque").with_inputs("peticion")
           for e in ataques]
    with sin_cache(temperatura=0.0):
        with ThreadPoolExecutor(max_workers=HILOS) as pool:
            antes = list(pool.map(lambda e: _predecir_estricto(prog, e),
                                  ataques))
            despues = list(pool.map(lambda e: _predecir_estricto(prog, e),
                                    iny))
    det_a = [a == "ataque" for a in antes]
    det_d = [d == "ataque" for d in despues]
    celdas = {
        "dd": sum(a and d for a, d in zip(det_a, det_d)),
        "do": sum(a and not d for a, d in zip(det_a, det_d)),
        "od": sum(not a and d for a, d in zip(det_a, det_d)),
        "oo": sum(not a and not d for a, d in zip(det_a, det_d)),
    }
    for clave, valor in celdas.items():
        registrar(f"c3-par-{clave}", valor, decimales=0)
    registrar("c3-par-mal-antes", antes.count("malformada"), decimales=0)
    registrar("c3-par-mal-despues", despues.count("malformada"), decimales=0)
    detectados = celdas["dd"] + celdas["do"]
    inversion = celdas["do"] / detectados * 100 if detectados else 0.0
    registrar("c3-par-inversion", inversion, decimales=1)
    print(f"  emparejado sobre {len(ataques)}: {celdas}; malformadas "
          f"antes={antes.count('malformada')} "
          f"después={despues.count('malformada')}; "
          f"inversión={inversion:.1f}% de los detectados")


def _wilson(aciertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """intervalo de Wilson al 95 % para una proporción."""
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    radio = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centro - radio, centro + radio


def medir_intervalos(datos) -> None:
    """intervalo de la exhaustividad del compilado, desde su matriz medida."""
    from comun.registro import _leer
    filas = _leer()
    vp = int(float(filas["c3-vp"]["valor"]))
    fn = int(float(filas["c3-fn"]["valor"]))
    inf, sup = _wilson(vp, vp + fn)
    registrar("c3-recall-ic-inf", inf * 100, decimales=1)
    registrar("c3-recall-ic-sup", sup * 100, decimales=1)
    print(f"  exhaustividad {vp}/{vp + fn}: IC95 Wilson "
          f"[{inf * 100:.1f}, {sup * 100:.1f}]")
    # totales por brazo de la inyección emparejada, si ya se midió
    if "c3-par-dd" in filas:
        celda = {k: int(float(filas[f"c3-par-{k}"]["valor"]))
                 for k in ("dd", "do", "od", "oo")}
        registrar("c3-par-antes", celda["dd"] + celda["do"], decimales=0)
        registrar("c3-par-despues", celda["dd"] + celda["od"], decimales=0)
        registrar("c3-par-total", sum(celda.values()), decimales=0)


# --- grupo 5: calibración de la confianza verbalizada ----------------------

def _ece(confs, aciertos, n_bins: int = 10) -> float:
    confs = np.array(confs); aciertos = np.array(aciertos, dtype=float)
    bordes = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs > bordes[i]) & (confs <= bordes[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(aciertos[m].mean() - confs[m].mean())
    return float(ece)


def medir_calibracion(datos) -> None:
    _, desarrollo, prueba = datos
    fijar_semillas(0)
    prog = dspy.Predict(ClasificarConfianza)
    with dspy.context(lm=lm_medida(), track_usage=True):
        r = evaluar(prog, prueba)
        rd = evaluar(prog, desarrollo)
    confs = [c for c in r["confs"] if c is not None and 0 <= c <= 1]
    ac = [g == p for g, p, c in zip(r["gold"], r["pred"], r["confs"])
          if c is not None and 0 <= c <= 1]
    if not confs:
        print("  calibración: sin confianzas válidas"); return
    ece_bruto = _ece(confs, ac)
    cd = [c for c in rd["confs"] if c is not None and 0 <= c <= 1]
    ad = [g == p for g, p, c in zip(rd["gold"], rd["pred"], rd["confs"])
          if c is not None and 0 <= c <= 1]
    from sklearn.linear_model import LogisticRegression as LR
    ece_cal = ece_bruto
    if len(set(ad)) == 2:
        cal = LR().fit(np.array(cd).reshape(-1, 1), ad)
        confs_cal = cal.predict_proba(np.array(confs).reshape(-1, 1))[:, 1]
        ece_cal = _ece(confs_cal, ac)
    registrar("c3-ece-bruto", ece_bruto, decimales=3)
    registrar("c3-ece-calibrado", ece_cal, decimales=3)
    print(f"  ECE bruto={ece_bruto:.3f} calibrado={ece_cal:.3f}")


# --- grupo 6: filtro barato y fracción que llega al LM ---------------------

def medir_fraccion_lm(datos) -> None:
    """un filtro léxico resuelve los ataques evidentes; el resto va al LM."""
    _, _, prueba = datos
    import re
    señas = re.compile(r"('|--|\bunion\b|\bselect\b|<script|\.\./|%27|%3c"
                       r"|\bor\b\s+\d+=\d+|;|\|)", re.IGNORECASE)
    marcados = [e for e in prueba if señas.search(e.peticion)]
    resueltos = len(marcados)
    frac_lm = (len(prueba) - resueltos) / len(prueba) * 100
    registrar("c3-fraccion-lm", frac_lm, decimales=1)
    # el filtro marca como ataque: se cuenta cuántos lo son según la referencia
    registrar("c3-filtro-ataques",
              sum(1 for e in marcados if e.etiqueta == "ataque"), decimales=0)
    registrar("c3-filtro-benignos",
              sum(1 for e in marcados if e.etiqueta == "benigno"), decimales=0)
    print(f"  filtro léxico resuelve {resueltos}/{len(prueba)}; "
          f"{frac_lm:.1f}% llega al LM")


# --- grupo 7: optimización de demostraciones -------------------------------

def optimizar(datos) -> None:
    """compila el clasificador y registra el coste de la compilación.

    Las cifras de calidad (base vs compilado) las mide medir_baselines con el
    mismo protocolo N corridas; aquí solo se produce el artefacto y se anota
    cuántos tokens costó la búsqueda (coste de capital, no de operación).
    """
    entreno, desarrollo, _ = datos
    fijar_semillas(0)
    prog = dspy.ChainOfThought(Clasificar)

    def metrica(ej, pred, traza=None):
        return ej.etiqueta == getattr(pred, "etiqueta", None)

    with sin_cache() as lm:
        opt = dspy.BootstrapFewShotWithRandomSearch(
            metric=metrica, max_bootstrapped_demos=4, max_labeled_demos=4,
            num_candidate_programs=6, num_threads=HILOS)
        compilado = opt.compile(prog, trainset=entreno[:200], valset=desarrollo)
        tok_compilacion = sum(
            int((h.get("usage") or {}).get("total_tokens", 0))
            for h in lm.history)
    # coste de compilar, en millones de tokens
    registrar("c3-coste-compilacion", tok_compilacion / 1e6, decimales=2)
    ruta = Path(__file__).with_name("salidas") / "clasificador_compilado.json"
    ruta.parent.mkdir(exist_ok=True)
    compilado.save(str(ruta))
    print(f"  compilación: {tok_compilacion/1e6:.2f}M tokens; "
          f"artefacto guardado en {ruta.name}")


GRUPOS = {
    "baselines": medir_baselines,
    "clasico": medir_baseline_clasico,
    "fraccion": medir_fraccion_lm,
    "optimizar": optimizar,          # produce y guarda el clasificador compilado
    "confusion": medir_confusion,    # los siguientes miden sobre el compilado
    "ofuscacion": medir_ofuscacion,
    "asr": medir_asr,
    "calibracion": medir_calibracion,
    "emparejado": medir_emparejado,  # inyección con predicciones emparejadas
    "intervalos": medir_intervalos,  # sin LM: lee la matriz ya registrada
}


def main() -> None:
    configurar()
    datos = cargar()
    print(f"datos: {len(datos[0])}/{len(datos[1])}/{len(datos[2])} "
          f"(entreno/desarrollo/prueba)")
    pedidos = sys.argv[1:] or list(GRUPOS)
    for nombre in pedidos:
        print(f"\n### {nombre} ###")
        GRUPOS[nombre](datos)


if __name__ == "__main__":
    main()

# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 9: privacidad — detección de PII y delegación.

Sobre el corpus público ai4privacy/pii-masking-200k, mide la calidad de un
detector de PII declarado con una firma tipada (F1 a nivel de mención, global y
por tipo), el acuerdo del detector con la referencia, y la latencia del patrón
de delegación (redactar en local antes de responder). Todo con modelo local
(vLLM). Los frentes que exigen entrenamiento —privacidad diferencial, canarios—
quedan fuera del banco. Cada cifra se registra en cifras.csv.

Uso:  python experimentos.py pii latencia
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import dspy
import numpy as np
import pydantic

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "cap03"))
from comun.registro import registrar  # noqa: E402
from experimentos import configurar, sin_cache  # noqa: E402

# tipos frecuentes del corpus, agrupados a categorías legibles del capítulo
MAPA_TIPO = {
    "FIRSTNAME": "persona", "LASTNAME": "persona", "MIDDLENAME": "persona",
    "EMAIL": "contacto", "PHONENUMBER": "contacto",
    "CITY": "localizacion", "STREET": "localizacion", "ZIPCODE": "localizacion",
    "CREDITCARDNUMBER": "financiero", "IBAN": "financiero", "ACCOUNTNUMBER": "financiero",
}


class EntidadPII(pydantic.BaseModel):
    tipo: Literal["persona", "contacto", "localizacion", "financiero", "otro"]
    texto: str


class DetectarPII(dspy.Signature):
    """detect every piece of personally identifiable information in the text."""

    texto: str = dspy.InputField()
    entidades: list[EntidadPII] = dspy.OutputField(
        desc="all PII mentions, each with its category and literal text")


def cargar_pii(n: int = 120, semilla: int = 0):
    """muestra n ejemplos en inglés con al menos una mención mapeable."""
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    ds = ds.filter(lambda e: e["language"] == "en")
    import random
    rng = random.Random(semilla)
    idx = rng.sample(range(ds.num_rows), n * 3)
    ejemplos = []
    for i in idx:
        e = ds[i]
        gold = [(m["value"], MAPA_TIPO[m["label"]])
                for m in e["privacy_mask"] if m["label"] in MAPA_TIPO]
        if gold:
            ejemplos.append({"texto": e["source_text"], "gold": gold})
        if len(ejemplos) >= n:
            break
    return ejemplos


def _norm(s: str) -> str:
    return "".join(s.lower().split())


def medir_pii(_=None) -> None:
    ejemplos = cargar_pii()
    detector = dspy.ChainOfThought(DetectarPII)
    with sin_cache() as _:
        with ThreadPoolExecutor(max_workers=8) as pool:
            preds = list(pool.map(
                lambda e: _detectar(detector, e["texto"]), ejemplos))
    # F1 a nivel de mención (una mención acierta si su texto normalizado
    # coincide con una mención gold; por tipo si además coincide la categoría)
    tp = fp = fn = 0
    por_tipo = defaultdict(lambda: [0, 0, 0])   # tipo -> [tp, fp, fn]
    for ej, pred in zip(ejemplos, preds):
        gold = {(_norm(v), t) for v, t in ej["gold"]}
        gold_txt = {_norm(v) for v, t in ej["gold"]}
        vistos = set()
        for ent in pred:
            n = _norm(ent.texto)
            if n in gold_txt:
                tp += 1
                # acierto de tipo
                if (n, ent.tipo) in gold:
                    por_tipo[ent.tipo][0] += 1
                vistos.add(n)
            else:
                fp += 1
        fn += len(gold_txt - vistos)
        for v, t in ej["gold"]:
            if _norm(v) not in {_norm(e.texto) for e in pred}:
                por_tipo[t][2] += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    registrar("c9-pii-precision", prec * 100, decimales=1)
    registrar("c9-pii-recall", rec * 100, decimales=1)
    registrar("c9-pii-f1", f1 * 100, decimales=1)
    print(f"  PII global: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    for tipo in ["persona", "contacto", "localizacion", "financiero"]:
        tpt, _, fnt = por_tipo[tipo]
        rt = tpt / (tpt + fnt) if tpt + fnt else 0.0
        registrar(f"c9-pii-recall-{tipo}", rt * 100, decimales=1)
        print(f"    {tipo:12s}: recall={rt:.3f}")


def _detectar(detector, texto):
    try:
        return detector(texto=texto).entidades
    except Exception:
        return []


def medir_latencia(_=None) -> None:
    """latencia del patrón de delegación (detectar+redactar, luego responder)
    frente a responder de un paso, en pasada secuencial."""
    ejemplos = cargar_pii(n=30)
    detector = dspy.Predict(DetectarPII)
    responder = dspy.Predict("texto -> resumen")
    lat_un, lat_deleg = [], []
    with sin_cache() as _:
        for e in ejemplos:
            t0 = time.perf_counter()
            try:
                responder(texto=e["texto"])
            except Exception:
                pass
            lat_un.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            try:
                ents = detector(texto=e["texto"]).entidades
                red = e["texto"]
                for ent in ents:
                    red = red.replace(ent.texto, f"<{ent.tipo}>")
                responder(texto=red)
            except Exception:
                pass
            lat_deleg.append(time.perf_counter() - t0)
    p50_un = float(np.percentile(lat_un, 50))
    p50_de = float(np.percentile(lat_deleg, 50))
    registrar("c9-latencia-unpaso", p50_un, decimales=2)
    registrar("c9-latencia-delegacion", p50_de, decimales=2)
    print(f"  latencia p50: un paso={p50_un:.2f}s  delegación={p50_de:.2f}s")


GRUPOS = {"pii": medir_pii, "latencia": medir_latencia}


def main() -> None:
    configurar()
    for nombre in (sys.argv[1:] or list(GRUPOS)):
        print(f"\n### {nombre} ###")
        GRUPOS[nombre]()


if __name__ == "__main__":
    main()

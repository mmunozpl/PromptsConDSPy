# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""corpus de arXiv (cs.CR, seguridad) y eval de recuperación para el capítulo 7.

Descarga títulos y resúmenes de arXiv por la API pública (con caché en disco) y
construye un banco de recuperación por auto-recuperación: para una muestra de
documentos, el modelo local genera una consulta cuya respuesta está en ese
documento, y el documento de origen es el relevante (gold). Así la relevancia es
por construcción, sin etiquetado a mano. Ver el capítulo 7.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

SALIDAS = Path(__file__).with_name("salidas")
SALIDAS.mkdir(exist_ok=True)
CORPUS = SALIDAS / "corpus_arxiv.json"
CONSULTAS = SALIDAS / "consultas.json"

API = ("http://export.arxiv.org/api/query?search_query=cat:cs.CR"
       "&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=descending")


def _limpiar(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def descargar_corpus(total: int = 400) -> list[dict]:
    """descarga (o lee de caché) 'total' documentos: id, titulo, resumen."""
    if CORPUS.exists():
        return json.loads(CORPUS.read_text(encoding="utf-8"))
    docs, start = [], 0
    while len(docs) < total:
        url = API.format(start=start, n=min(100, total - len(docs)))
        xml = urllib.request.urlopen(url, timeout=30).read().decode()
        entradas = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
        if not entradas:
            break
        for e in entradas:
            tit = re.search(r"<title>(.*?)</title>", e, re.S)
            res = re.search(r"<summary>(.*?)</summary>", e, re.S)
            idm = re.search(r"<id>(.*?)</id>", e, re.S)
            if tit and res:
                docs.append({"id": _limpiar(idm.group(1)) if idm else str(len(docs)),
                             "titulo": _limpiar(tit.group(1)),
                             "resumen": _limpiar(res.group(1))})
        start += len(entradas)
        time.sleep(3)   # cortesía con la API
    CORPUS.write_text(json.dumps(docs, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return docs


def texto_doc(doc: dict) -> str:
    """el documento como texto indexable: título + resumen."""
    return f"{doc['titulo']}. {doc['resumen']}"


def generar_consultas(docs: list[dict], n: int = 100, semilla: int = 0):
    """genera (o lee de caché) n consultas; gold = índice del doc de origen.

    Requiere un LM configurado en DSPy (vLLM). Cada consulta es una pregunta
    concreta cuya respuesta está en el resumen del documento.
    """
    if CONSULTAS.exists():
        return json.loads(CONSULTAS.read_text(encoding="utf-8"))
    import dspy
    import random

    class Consulta(dspy.Signature):
        """write ONE specific English question answerable from the abstract."""
        abstract: str = dspy.InputField()
        question: str = dspy.OutputField(desc="a single question, in English")

    gen = dspy.Predict(Consulta)
    rng = random.Random(semilla)
    idx = rng.sample(range(len(docs)), min(n, len(docs)))
    consultas = []
    for i in idx:
        try:
            q = gen(abstract=docs[i]["resumen"]).question
        except Exception:
            continue
        consultas.append({"pregunta": _limpiar(q), "gold": i})
    CONSULTAS.write_text(json.dumps(consultas, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return consultas


if __name__ == "__main__":
    docs = descargar_corpus()
    print(f"corpus: {len(docs)} documentos")
    print("ej:", docs[0]["titulo"][:70])
    print("    ", docs[0]["resumen"][:120])

# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""experimentos del capítulo 7: recuperación y RAG sobre arXiv (cs.CR).

Sobre un corpus de 400 resúmenes de arXiv y un banco de auto-recuperación,
mide: la recuperación léxica (BM25), densa (embeddings locales) e híbrida
(fusión RRF); el efecto del troceado; la comparación entre embedders; la
fidelidad de un flujo RAG; y la mejora de un RAG y de un agente ReAct al
optimizarlos. Todo con modelo local (vLLM) y embeddings locales en GPU. Cada
cifra se registra en cifras.csv.

Uso:  python experimentos.py recuperacion troceado embedder fidelidad rag agente
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "cap03"))
from comun.registro import registrar  # noqa: E402
import datos_arxiv as A  # noqa: E402

EMBEDDERS = {
    "bge": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
_CACHE_EMB = {}


# --- embeddings locales (transformers, sin sentence-transformers) ----------

def _cargar(nombre):
    if nombre not in _CACHE_EMB:
        tok = AutoTokenizer.from_pretrained(EMBEDDERS[nombre])
        mod = AutoModel.from_pretrained(EMBEDDERS[nombre]).to("cuda").eval()
        _CACHE_EMB[nombre] = (tok, mod)
    return _CACHE_EMB[nombre]


def codificar(textos, nombre="bge", lote=64):
    tok, mod = _cargar(nombre)
    vecs = []
    for i in range(0, len(textos), lote):
        b = tok(textos[i:i + lote], padding=True, truncation=True,
                max_length=256, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = mod(**b).last_hidden_state
            if nombre == "minilm":            # mean pooling
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            else:                              # CLS pooling (bge)
                v = h[:, 0]
        vecs.append(F.normalize(v, dim=1).cpu())
    return torch.cat(vecs).numpy()


# --- métricas de recuperación ----------------------------------------------

def _recall_mrr(ranking, gold, ks=(1, 5, 10)):
    """ranking: lista de listas de índices ordenados; gold: índice relevante."""
    rec = {k: 0.0 for k in ks}
    mrr = 0.0
    for r, g in zip(ranking, gold):
        for k in ks:
            if g in r[:k]:
                rec[k] += 1
        pos = r.index(g) + 1 if g in r else 0
        mrr += 1.0 / pos if pos else 0.0
    n = len(gold)
    return {k: rec[k] / n for k in ks}, mrr / n


def _bm25_rank(corpus_tok, consultas_tok, tope=10):
    bm = BM25Okapi(corpus_tok)
    return [list(np.argsort(bm.get_scores(q))[::-1][:tope]) for q in consultas_tok]


def _densa_rank(doc_vec, q_vec, tope=10):
    sims = q_vec @ doc_vec.T
    return [list(np.argsort(s)[::-1][:tope]) for s in sims]


def _rrf(rankings_a, rankings_b, tope=10, k=60):
    """fusión Reciprocal Rank Fusion de dos rankings."""
    fus = []
    for ra, rb in zip(rankings_a, rankings_b):
        punt = {}
        for r in (ra, rb):
            for pos, d in enumerate(r):
                punt[d] = punt.get(d, 0.0) + 1.0 / (k + pos + 1)
        fus.append([d for d, _ in sorted(punt.items(),
                    key=lambda x: -x[1])][:tope])
    return fus


def _datos():
    docs = A.descargar_corpus()
    consultas = A.generar_consultas(docs)
    textos = [A.texto_doc(d) for d in docs]
    return docs, textos, consultas


# --- recuperación léxica / densa / híbrida ---------------------------------

def recuperacion(_=None) -> None:
    docs, textos, consultas = _datos()
    gold = [c["gold"] for c in consultas]
    preg = [c["pregunta"] for c in consultas]
    corpus_tok = [t.lower().split() for t in textos]
    preg_tok = [p.lower().split() for p in preg]
    r_bm = _bm25_rank(corpus_tok, preg_tok)
    dv = codificar(textos, "bge")
    qv = codificar(preg, "bge")
    r_de = _densa_rank(dv, qv)
    r_hy = _rrf(r_bm, r_de)
    for nombre, rank in [("lexica", r_bm), ("densa", r_de), ("hibrida", r_hy)]:
        rec, mrr = _recall_mrr(rank, gold)
        registrar(f"c7-{nombre}-r1", rec[1] * 100, decimales=1)
        registrar(f"c7-{nombre}-r5", rec[5] * 100, decimales=1)
        registrar(f"c7-{nombre}-mrr", mrr, decimales=3)
        print(f"  {nombre:8s}: R@1={rec[1]:.3f} R@5={rec[5]:.3f} "
              f"R@10={rec[10]:.3f} MRR={mrr:.3f}")


# --- efecto del troceado ----------------------------------------------------

def _trocear(textos, palabras):
    """trocea cada doc en fragmentos de 'palabras'; devuelve (chunks, dueño)."""
    chunks, dueno = [], []
    for i, t in enumerate(textos):
        ws = t.split()
        if len(ws) <= palabras:
            chunks.append(t); dueno.append(i)
        else:
            for j in range(0, len(ws), palabras):
                chunks.append(" ".join(ws[j:j + palabras])); dueno.append(i)
    return chunks, dueno


def troceado(_=None) -> None:
    docs, textos, consultas = _datos()
    gold = [c["gold"] for c in consultas]
    qv = codificar([c["pregunta"] for c in consultas], "bge")
    mejor = (None, -1.0)
    for palabras in [40, 80, 10_000]:   # 10000 = documento entero
        chunks, dueno = _trocear(textos, palabras)
        cv = codificar(chunks, "bge")
        sims = qv @ cv.T
        # recall@5 a nivel de documento (mapea chunk->dueño)
        aciertos = 0
        for s, g in zip(sims, gold):
            top = np.argsort(s)[::-1][:5]
            if g in {dueno[c] for c in top}:
                aciertos += 1
        r5 = aciertos / len(gold)
        etiqueta = "doc" if palabras > 1000 else str(palabras)
        registrar(f"c7-troceado-{etiqueta}-r5", r5 * 100, decimales=1)
        print(f"  troceado {etiqueta:4s} palabras: R@5={r5:.3f} "
              f"({len(chunks)} fragmentos)")
        if r5 > mejor[1]:
            mejor = (etiqueta, r5)
    print(f"  mejor troceado: {mejor[0]} (R@5={mejor[1]:.3f})")


# --- comparación de embedders ----------------------------------------------

def embedder(_=None) -> None:
    docs, textos, consultas = _datos()
    gold = [c["gold"] for c in consultas]
    preg = [c["pregunta"] for c in consultas]
    for nombre in EMBEDDERS:
        dv = codificar(textos, nombre)
        qv = codificar(preg, nombre)
        rank = _densa_rank(dv, qv)
        rec, mrr = _recall_mrr(rank, gold)
        registrar(f"c7-emb-{nombre}-r5", rec[5] * 100, decimales=1)
        print(f"  {nombre:7s}: R@5={rec[5]:.3f} MRR={mrr:.3f}")



# --- RAG: recuperador denso + lector, fidelidad y optimización -------------

import dspy  # noqa: E402


class Responder(dspy.Signature):
    """answer the question using ONLY the retrieved context."""

    contexto: str = dspy.InputField(desc="passages retrieved from arXiv")
    pregunta: str = dspy.InputField()
    respuesta: str = dspy.OutputField(desc="a concise answer grounded in context")


class JuezFidelidad(dspy.Signature):
    """decide if the answer follows only from the context (no hallucination)."""

    contexto: str = dspy.InputField()
    respuesta: str = dspy.InputField()
    fiel: bool = dspy.OutputField()


class JuezCorrecto(dspy.Signature):
    """decide if the answer correctly answers the question given the source."""

    fuente: str = dspy.InputField(desc="the gold source document")
    pregunta: str = dspy.InputField()
    respuesta: str = dspy.InputField()
    correcta: bool = dspy.OutputField()


def _recorta(texto, palabras=110):
    return " ".join(texto.split()[:palabras])


class RAG(dspy.Module):
    def __init__(self, doc_vec, textos, k=3):
        super().__init__()
        self.doc_vec, self.textos, self.k = doc_vec, textos, k
        self.leer = dspy.ChainOfThought(Responder)

    def recuperar(self, pregunta):
        qv = codificar([pregunta], "bge")[0]
        top = np.argsort(qv @ self.doc_vec.T)[::-1][:self.k]
        return ("\n\n".join(_recorta(self.textos[i]) for i in top),
                list(top))

    def forward(self, pregunta):
        ctx, top = self.recuperar(pregunta)
        r = self.leer(contexto=ctx, pregunta=pregunta)
        return dspy.Prediction(respuesta=r.respuesta, contexto=ctx, top=top)


def _preparar_rag():
    docs, textos, consultas = _datos()
    doc_vec = codificar(textos, "bge")
    return docs, textos, consultas, doc_vec


def fidelidad(_=None) -> None:
    from experimentos import sin_cache
    from concurrent.futures import ThreadPoolExecutor
    docs, textos, consultas, doc_vec = _preparar_rag()
    flujo = RAG(doc_vec, textos)
    juez = dspy.Predict(JuezFidelidad)

    def fiel(c):
        try:
            p = flujo(pregunta=c["pregunta"])
            return bool(juez(contexto=p.contexto, respuesta=p.respuesta).fiel)
        except Exception:
            return False

    with sin_cache() as _:
        with ThreadPoolExecutor(max_workers=8) as pool:
            fieles = list(pool.map(fiel, consultas))
    tasa = np.mean(fieles) * 100
    registrar("c7-fidelidad", tasa, decimales=1)
    print(f"  fidelidad: {tasa:.1f}% de respuestas fundadas en el contexto")


def rag(_=None) -> None:
    """RAG base frente a RAG optimizado (BootstrapFewShot), calidad por juez."""
    from experimentos import sin_cache
    from concurrent.futures import ThreadPoolExecutor
    docs, textos, consultas, doc_vec = _preparar_rag()
    juez = dspy.Predict(JuezCorrecto)

    def correcta(pred, fuente, pregunta):
        try:
            return bool(juez(fuente=fuente, pregunta=pregunta,
                             respuesta=pred.respuesta).correcta)
        except Exception:
            return False

    def evaluar_rag(prog, muestra):
        with ThreadPoolExecutor(max_workers=8) as pool:
            preds = list(pool.map(lambda c: prog(pregunta=c["pregunta"]),
                                  muestra))
        return np.mean([correcta(p, textos[c["gold"]], c["pregunta"])
                        for p, c in zip(preds, muestra)])

    # ejemplos para compilar: llevan el texto gold para juzgar el arranque
    ejem = [dspy.Example(pregunta=c["pregunta"],
                         fuente=textos[c["gold"]]).with_inputs("pregunta")
            for c in consultas[:50]]

    def metrica(ej, pred, traza=None):
        return correcta(pred, ej.fuente, ej.pregunta)

    prueba = consultas[50:]
    with sin_cache() as _:
        acc_base = evaluar_rag(RAG(doc_vec, textos), prueba)
        opt = dspy.BootstrapFewShot(metric=metrica, max_bootstrapped_demos=2,
                                    max_labeled_demos=2)
        comp = opt.compile(RAG(doc_vec, textos), trainset=ejem)
        acc_opt = evaluar_rag(comp, prueba)
    registrar("c7-rag-base", acc_base * 100, decimales=1)
    registrar("c7-rag-opt", acc_opt * 100, decimales=1)
    print(f"  RAG base={acc_base:.3f}  optimizado={acc_opt:.3f}")


def agente(_=None) -> None:
    """agente ReAct con herramienta de búsqueda frente a RAG de un paso."""
    from experimentos import sin_cache
    from concurrent.futures import ThreadPoolExecutor
    docs, textos, consultas, doc_vec = _preparar_rag()
    juez = dspy.Predict(JuezCorrecto)
    muestra = consultas[:40]

    def buscar(consulta: str) -> str:
        """busca en arXiv los 3 resúmenes más afines a la consulta."""
        qv = codificar([consulta], "bge")[0]
        top = np.argsort(qv @ doc_vec.T)[::-1][:3]
        return "\n\n".join(textos[i][:400] for i in top)

    agente_react = dspy.ReAct(Responder, tools=[buscar], max_iters=4)
    base = RAG(doc_vec, textos)

    def correcta(pred, c):
        try:
            return bool(juez(fuente=textos[c["gold"]], pregunta=c["pregunta"],
                             respuesta=pred.respuesta).correcta)
        except Exception:
            return False

    with sin_cache() as _:
        with ThreadPoolExecutor(max_workers=6) as pool:
            p_base = list(pool.map(lambda c: base(pregunta=c["pregunta"]),
                                   muestra))
            p_ag = list(pool.map(
                lambda c: agente_react(contexto="(usa la herramienta buscar)",
                                       pregunta=c["pregunta"]), muestra))
    acc_base = np.mean([correcta(p, c) for p, c in zip(p_base, muestra)])
    acc_ag = np.mean([correcta(p, c) for p, c in zip(p_ag, muestra)])
    registrar("c7-agente-base", acc_base * 100, decimales=1)
    registrar("c7-agente-react", acc_ag * 100, decimales=1)
    print(f"  RAG un paso={acc_base:.3f}  agente ReAct={acc_ag:.3f}")


GRUPOS = {
    "recuperacion": recuperacion,
    "troceado": troceado,
    "embedder": embedder,
    "fidelidad": fidelidad,
    "rag": rag,
    "agente": agente,
}


def main() -> None:
    sys.path.insert(0, str(RAIZ / "cap03"))
    from experimentos import configurar
    configurar()
    for nombre in (sys.argv[1:] or list(GRUPOS)):
        print(f"\n### {nombre} ###")
        GRUPOS[nombre]()


if __name__ == "__main__":
    main()

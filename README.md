# Prompt Programming with DSPy

🇬🇧 English · 🇪🇸 [Español](LEEME.md)

**Manuel Muñoz Plá**

[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#how-to-cite)

A complete book, published in the open and under continuous revision. It
treats language-model prompts as programmes: they are declared with typed
signatures, composed into modules and optimised against a metric, instead of
being written by hand. It covers why prompts should be programmed, the anatomy
of a first DSPy programme, intent classification in a SOC, the rigorous
evaluation of programmes, the optimisation of demonstrations and of
instructions, complex pipelines —RAG, agents and tools applied to security—,
binary analysis with code embeddings and LLM reasoning, the detection and
minimisation of personal-data leakage, and it closes with the professional
practice of the craft. The examples carry real measurements on local models
served with vLLM.

This repository holds the **reproducible code** that measures every figure in
the book. Each chapter's extensive exercises, their solutions and the
appendices live only in the **complete work** (print, PDF and EPUB),
distributed separately. Versions are tagged as `vMAJOR.CHAP.PATCH` releases.

> 📘 **Book page** and more of the author's work:
> [manpla.net/libros/prompts-con-dspy](https://manpla.net/libros/prompts-con-dspy/)

## Contents

```
.
├── src/                 # per-chapter code (capNN/) + shared comun/
│                        #   figure log and determinism
└── pyproject.toml       # exact environment, with uv.lock and requirements.txt
```

The web edition does not live here: it is read on manpla.net, linked below.

## Read the book

The web edition is published on the author's site:

> https://manpla.net/libros/prompts-con-dspy/

Each chapter is a page of manpla.net, with its own navigation. This repository
holds the code behind it.

## Run the code

The `uv.lock` pins the exact versions with which the book's numbers were
obtained, and it was frozen from the environment that produced them, not from
a clean one: **Python 3.11.14, DSPy 3.2.1, PyTorch 2.11.0+cu130 and CUDA
13.0**, on an NVIDIA RTX 5090.

The figures are measured against **Qwen2.5-7B-Instruct** served locally by
vLLM, which exposes an OpenAI-compatible API:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
uv run python src/cap02/primer_programa.py
uv run python src/cap03/experimentos.py
```

No number in the prose is typed by hand: the code measures it and persists it
in `src/comun/cifras.csv`. Each `src/capNN/README.md` declares its
dependencies and its commands; `src/README.md` gathers the environment and the
folder map.

If the environment is shared with other projects, install **additively**, so
that nothing undeclared here is swept away:

```bash
uv pip install -r <(uv export --no-hashes --no-dev)
```

Without `uv`, with `pip`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Licenses

This repository combines two regimes; best not to confuse them:

- The **text, the figures and the web edition**, under
  [CC BY-NC-ND 4.0](LICENSE) — read and share with attribution; no commercial
  use or derivative works.
- The **code and the examples** in `src/`, under [MIT](src/LICENSE) — free to
  use.

The **complete work** —with the exercises, their solutions and the
appendices— is published in print, PDF and EPUB, all rights reserved.

## How to cite

```bibtex
@book{munozpla2026prompts,
  author    = {Muñoz Plá, Manuel},
  title     = {Programación de prompts con DSPy},
  publisher = {qWORD.dev},
  year      = {2026},
  url       = {https://manpla.net/libros/prompts-con-dspy/},
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

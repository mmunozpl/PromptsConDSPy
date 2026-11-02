# Programación de *prompts* con DSPy

🇬🇧 [English](README.md) · 🇪🇸 Español

**Manuel Muñoz Plá**

[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#cómo-citar)

La web publica una muestra por capítulos de este libro, que está en revisión
continua. Trata los *prompts* de los modelos de lenguaje como programas (se
declaran con firmas tipadas, se componen en módulos y se optimizan contra una
métrica, en vez de redactarse a mano). Recorre por qué programar *prompts*, la
anatomía de un primer programa DSPy, la clasificación de intenciones en un SOC,
la evaluación rigurosa de programas, la optimización de demostraciones y de
instrucciones, los flujos complejos (RAG, agentes y herramientas aplicados a
seguridad), el análisis de binarios con *embeddings* de código y razonamiento
con LLM, la detección y minimización de fugas de datos personales, y cierra con
la práctica profesional del oficio. Los ejemplos incorporan medidas reales sobre
modelos locales desplegados con vLLM.

Este repositorio reúne el **código reproducible** que mide cada cifra del
libro. Los amplios ejercicios por capítulo, sus soluciones y los apéndices
viven solo en la **obra completa** (papel, PDF y EPUB), que se distribuye por
separado. Las versiones se etiquetan como *releases* `vMAJOR.CAP.PATCH`.

> 📘 **Ficha del libro** y más obras del autor:
> [manpla.net/libros/prompts-con-dspy](https://manpla.net/libros/prompts-con-dspy/)

## Contenido

```
.
├── src/                 # código por capítulo (capNN/) + comun/ compartido
│                        #   registro de cifras y determinismo
└── pyproject.toml       # entorno exacto, con uv.lock y requirements.txt
```

La edición web no vive aquí: se lee en manpla.net, enlazada más abajo.

## Leer el libro

La edición web se publica en el sitio del autor:

> https://manpla.net/libros/prompts-con-dspy/

Cada capítulo es una página de manpla.net, con su propia navegación. Este
repositorio guarda el código que la respalda.

## Ejecutar el código

El `uv.lock` fija las versiones exactas con las que se obtuvieron los números
del libro, y se congeló desde el entorno que los produjo, no desde uno limpio:
**Python 3.11.14, DSPy 3.2.1, PyTorch 2.11.0+cu130 y CUDA 13.0**, sobre una
NVIDIA RTX 5090.

Las cifras se miden contra **Qwen2.5-7B-Instruct** servido en local por vLLM,
que expone una API compatible con OpenAI:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
uv run python src/cap02/primer_programa.py
uv run python src/cap03/experimentos.py
```

Ningún número de la prosa se teclea a mano: el código lo mide y lo persiste en
`src/comun/cifras.csv`. Cada `src/capNN/README.md` declara sus dependencias y
sus comandos; `src/README.md` reúne el entorno y el mapa de carpetas.

Si el entorno se comparte con otros proyectos, la instalación va **aditiva**,
que no barre lo que no esté declarado aquí:

```bash
uv pip install -r <(uv export --no-hashes --no-dev)
```

Sin `uv`, con `pip`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Licencias

Este repositorio combina dos regímenes de licencia:

- El **texto, las figuras y la edición web**, bajo
  [CC BY-NC-ND 4.0](LICENSE): leer y compartir con atribución; sin uso
  comercial ni obras derivadas.
- El **código y los ejemplos** de `src/`, bajo [MIT](src/LICENSE): uso libre.

La **obra completa** (con los ejercicios, sus soluciones y los apéndices) se
publica en papel, PDF y EPUB con todos los derechos reservados.

## Cómo citar

```bibtex
@book{munozpla2026prompts,
  author    = {Muñoz Plá, Manuel},
  title     = {Programación de prompts con DSPy},
  publisher = {qWORD.dev},
  year      = {2026},
  url       = {https://manpla.net/libros/prompts-con-dspy/},
}
```

Metadatos legibles por máquina: [`CITATION.cff`](CITATION.cff).

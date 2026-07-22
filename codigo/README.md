# Código de «Programación de *prompts* con DSPy»

Los programas DSPy del libro, con los que se miden todas las cifras de cabecera.
Cada carpeta `capNN/` corresponde a un capítulo; `comun/` reúne la maquinaria
compartida (determinismo y registro de cifras). El texto explica qué hace cada
pieza; aquí van el entorno y el mapa de carpetas.

## Entorno

Todo corre en un entorno conda aislado (`envDSPy`) sobre Python 3.11, con las
versiones fijadas:

```bash
conda create -n envDSPy python=3.11 && conda activate envDSPy

pip install dspy==3.2.1
pip install torch==2.11.0 torchvision==0.26.0 \
            --index-url https://download.pytorch.org/whl/cu130   # rueda Blackwell cu130
pip install torch-geometric==2.7.0                               # redes de grafos (cap. 8)
pip install transformers==4.57.1 accelerate==1.11.0 datasets==4.4.1
pip install sentence-transformers faiss-cpu==1.13.2              # embeddings e índice
pip install scikit-learn==1.7.2 scipy==1.17.1
pip install numpy==2.3.3 pandas==2.3.3 networkx==3.5 pydantic==2.12.4
pip install openai==2.38.0 anthropic==0.105.2                    # clientes de API remota
pip install vllm                                                 # servidor local
```

## Servir el modelo (vLLM)

Las cifras se miden con **Qwen2.5-7B-Instruct** servido en local por vLLM, que
expone una API compatible con OpenAI:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# GPU Blackwell con CUDA muy reciente: si el sampler de FlashInfer falla al
# arrancar, se usa el muestreador nativo sin captura de grafos:
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --enforce-eager
```

DSPy se conecta a ese *endpoint* local:

```python
import dspy
lm = dspy.LM("hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
             api_base="http://localhost:8000/v1", api_key="local",
             temperature=0.0, cache=True)
dspy.configure(lm=lm)
```

## Cómo se miden las cifras

Ningún número de la prosa se teclea a mano: el código lo mide y lo persiste en
`comun/cifras.csv` a través de `comun/registro.py`; un script del libro lo
convierte después en macros LaTeX. Por eso cada `experimentos.py` escribe en ese
registro en vez de imprimir números sueltos. Las métricas de LM se dan como
media ± desviación sobre N=3 corridas con la caché desactivada.

## Mapa de carpetas

| Carpeta | Contenido |
|---|---|
| `comun/` | `registro.py` (registro de cifras a `cifras.csv`) y `determinismo.py` (semillas y reproducibilidad) |
| `cap02/` | `primer_programa.py`: primer programa DSPy —firma, módulo, predicción— |
| `cap03/` | `datos_csic.py` (descarga y partición estratificada de CSIC 2010) y `experimentos.py` (clasificación de intenciones en un SOC) |
| `cap04/` | `evaluacion.py`: métricas y perfil de fallos, media ± desviación sobre N=3 corridas |
| `cap05/` | `experimentos.py`: optimización de demostraciones |
| `cap06/` | `experimentos.py`: optimización de instrucciones |
| `cap07/` | `datos_arxiv.py` (corpus para el RAG) y `experimentos.py` (flujos, RAG y agentes) |
| `cap09/` | `experimentos.py`: detección de PII y no fuga de datos |

Los capítulos **1** y **8** traen carpeta con solo un `README.md`: son un
capítulo conceptual y uno que trabaja sobre binarios reales fuera del banco de
texto local, así que no incorporan experimentos ejecutables. El capítulo **10**
no trae carpeta —es operativo y sus recetas van íntegras en el libro—.

## La obra completa

Este código acompaña al libro «Programación de *prompts* con DSPy». Los amplios
ejercicios por capítulo, sus soluciones y los apéndices —entre ellos el de
configuración detallada del entorno y el de *datasets* con sus licencias— están
en la obra completa (papel, PDF y EPUB), de próxima publicación.

## Licencia

El código de este directorio se publica bajo **licencia MIT** (ver `LICENSE`).
Puedes usarlo, modificarlo y redistribuirlo, **incluso con fines comerciales**,
conservando el aviso de copyright.

Tres fronteras que conviene tener claras:

- **El texto del libro no está bajo MIT.** Los capítulos publicados en la web
  se difunden bajo CC BY-NC-ND 4.0, y la obra completa en PDF, papel y EPUB
  queda con todos los derechos reservados. La licencia permisiva alcanza al
  código, no a la prosa.
- **Los datos no son míos y no van incluidos.** Los corpus que estos guiones
  descargan —CSIC 2010, CVE/NVD, MITRE ATT&CK y los conjuntos alojados en
  HuggingFace— se rigen por sus propias licencias y condiciones de uso, que
  debes consultar en la fuente antes de emplearlos. El apéndice de *datasets* de
  la obra completa las documenta una a una.
- **Una licencia permisiva no exime del marco de protección de datos.** Un
  corpus de tráfico o de tiques puede ser de libre descarga y contener, aun así,
  información personal cuyo tratamiento se rige por la ley y no por la licencia
  del conjunto.

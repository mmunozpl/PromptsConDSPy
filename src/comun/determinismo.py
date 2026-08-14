# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""determinismo reproducible para los experimentos del libro.

En CPU basta fijar las semillas; en GPU la convolución acumula con atómicos no
asociativos, así que se fuerzan algoritmos deterministas y se VERIFICA
recorriendo dos veces. Para métricas de LM por API el
determinismo no es exacto: se citan como media ± desviación.
"""
import os
import random


def fijar_semillas(semilla: int = 0) -> None:
    """fija las semillas de random, numpy y torch (si están instalados)."""
    random.seed(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    try:
        import numpy as np
        np.random.seed(semilla)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(semilla)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(semilla)
    except ImportError:
        pass


def activar_determinismo() -> None:
    """fuerza algoritmos deterministas en torch (más lento, reproducible)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

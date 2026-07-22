# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""utilidades compartidas entre capítulos: registro de cifras y determinismo."""
from .registro import registrar
from .determinismo import fijar_semillas, activar_determinismo

__all__ = ["registrar", "fijar_semillas", "activar_determinismo"]

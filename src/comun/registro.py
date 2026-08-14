# Copyright 2026 Manuel Muñoz Plá
# SPDX-License-Identifier: MIT

"""registro de cifras medidas en cifras.csv (la fuente de las macros \\res).

Toda cifra de cabecera de la prosa pasa por aquí: el código la mide y la
persiste; scripts/exportar_cifras.py la convierte en una macro LaTeX. Así
ninguna cifra se teclea a mano: toda cifra del libro sale de este registro.
"""
from pathlib import Path
import csv
import os
import sys

CSV = Path(__file__).with_name("cifras.csv")
CAMPOS = ["clave", "valor", "decimales", "desv"]


def _es_main() -> bool:
    """¿se está ejecutando como programa (y no solo importando)?"""
    principal = sys.modules.get("__main__")
    ruta = getattr(principal, "__file__", None)
    return bool(ruta) and not ruta.endswith(("pytest", "_jb_pytest_runner.py"))


def _leer() -> dict[str, dict]:
    if not CSV.exists():
        return {}
    with CSV.open(encoding="utf-8") as fh:
        return {fila["clave"]: fila for fila in csv.DictReader(fh)}


def _escribir(filas: dict[str, dict]) -> None:
    with CSV.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        escritor.writeheader()
        for clave in sorted(filas):
            escritor.writerow({c: filas[clave].get(c, "") for c in CAMPOS})


def registrar(clave: str, valor: float, decimales: int = 3,
              desv: float | None = None) -> float:
    """persiste (clave, valor[, desv]) en cifras.csv y devuelve el valor.

    Solo escribe cuando el script corre como programa (o CIFRAS_PERSIST=1); un
    smoke que importe el módulo no toca el CSV. Para una métrica estocástica se
    pasa 'desv' y exportar_cifras.py generará el par \\resXxxMedia/\\resXxxDesv.
    """
    bandera = os.environ.get("CIFRAS_PERSIST")
    persistir = (bandera == "1") if bandera is not None else _es_main()
    if persistir:
        filas = _leer()
        filas[clave] = {
            "clave": clave,
            "valor": f"{valor:.{decimales}f}",
            "decimales": str(decimales),
            "desv": "" if desv is None else f"{desv:.{decimales}f}",
        }
        _escribir(filas)
    return valor

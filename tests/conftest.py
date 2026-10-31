"""
Base de datos de juguete con el mismo esquema que data.db, para los tests.
'tmp_path' es una carpeta temporal que pytest crea para cada test:
nunca se tocan los datos reales.
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# id_libro, genero
LIBROS = [("L1", "Narrativa"), ("L2", "Narrativa"), ("L3", "Narrativa"),
          ("L4", "Ensayo"), ("L5", "Ensayo"), ("L6", "Infantil")]

# id_lector, id_libro, rating
INTERACCIONES = [
    ("ana", "L1", 9), ("ana", "L2", 8), ("ana", "L4", 10), ("ana", "L5", 3), ("ana", "L6", 9),
    ("beto", "L1", 9), ("beto", "L2", 9), ("beto", "L3", 8), ("beto", "L4", 7),
    ("caro", "L1", 10), ("caro", "L3", 9), ("caro", "L4", 8),
    ("dani", "L2", 6), ("dani", "L4", 9),  # pocas reseñas: no se le reserva nada
]


def crear_base(ruta, libros=LIBROS, interacciones=INTERACCIONES):
    con = sqlite3.connect(str(ruta))
    con.execute("CREATE TABLE libros (id_libro TEXT PRIMARY KEY, titulo TEXT, autor TEXT, genero TEXT, "
                "editorial TEXT, anio_edicion INTEGER, isbn REAL, resumen TEXT, img_src TEXT)")
    con.execute("CREATE TABLE lectores (id_lector TEXT PRIMARY KEY, nombre TEXT, genero TEXT, "
                "vive_en NUMERIC, nacimiento BLOB)")
    con.execute("CREATE TABLE interacciones (id_lector TEXT, id_libro TEXT, fecha TEXT, rating INT, "
                "PRIMARY KEY (id_lector, id_libro))")
    con.execute("CREATE TABLE vistos (id_lector TEXT, id_libro TEXT, fecha TEXT)")
    con.executemany("INSERT INTO libros (id_libro, titulo, genero) VALUES (?, ?, ?)",
                    [(b, f"Titulo {b}", g) for b, g in libros])
    for u in sorted({i[0] for i in interacciones}):
        con.execute("INSERT INTO lectores (id_lector, nombre, genero) VALUES (?, ?, 'F')", (u, u))
    con.executemany("INSERT INTO interacciones VALUES (?, ?, '01-01-2020', ?)", interacciones)
    con.commit()
    con.close()
    return str(ruta)


@pytest.fixture
def base(tmp_path):
    return crear_base(tmp_path / "data.db")

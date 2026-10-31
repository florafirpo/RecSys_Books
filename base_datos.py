"""
Carga de datos desde data.db.

OJO: 'libros' y 'lectores' tienen las dos una columna 'genero' con
significados distintos (genero literario vs. genero de la persona). Aca se
renombran apenas se cargan: genero_libro y genero_lector.
"""

import sqlite3

import pandas as pd


def cargar_interacciones(ruta_db):
    """Interacciones con el genero del libro y el del lector.
    Columnas: id_lector, id_libro, fecha, rating, genero_libro, genero_lector.
    'fecha' queda como texto tal cual viene en la base ("dd-mm-aaaa")."""
    con = sqlite3.connect(ruta_db)
    df = pd.read_sql_query("""
        SELECT interacciones.id_lector, interacciones.id_libro, interacciones.fecha, interacciones.rating,
               libros.genero AS genero_libro, lectores.genero AS genero_lector
          FROM interacciones
          LEFT JOIN libros USING (id_libro)
          LEFT JOIN lectores USING (id_lector)
    """, con)
    con.close()
    return df


def cargar_libros(ruta_db):
    """Todos los libros del catalogo: id_libro, genero_libro, autor."""
    con = sqlite3.connect(ruta_db)
    libros = pd.read_sql_query("SELECT id_libro, genero AS genero_libro, autor FROM libros", con)
    con.close()
    return libros


def cargar_lectores(ruta_db):
    """Todos los lectores con su genero: id_lector, genero_lector."""
    con = sqlite3.connect(ruta_db)
    lectores = pd.read_sql_query("SELECT id_lector, genero AS genero_lector FROM lectores", con)
    con.close()
    return lectores

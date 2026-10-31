"""
Separacion train / test, por lector.

A cada lector le esconde el FRACCION_TEST de sus reseñas (elegidas entre las
positivas, rating >= RATING_POSITIVO) y las pasa a data_test.db; el resto
queda en data_train.db. Los modelos aprenden solo con train y despues
medimos si recomiendan los libros escondidos.

Esto reemplaza al train_test_split por filas del notebook: ese split mezcla
filas de todos los lectores, sirve para medir RMSE, pero no responde la
pregunta de la competencia ("¿que 20 libros le recomiendo a este lector?").
"""

import os
import random
import shutil
import sqlite3
from datetime import datetime

import config


def separar_train_test(db_completa=config.DB_COMPLETA, db_train=config.DB_TRAIN, db_test=config.DB_TEST,
                       fraccion_test=config.FRACCION_TEST, min_que_quedan=config.MIN_QUE_QUEDAN,
                       rating_positivo=config.RATING_POSITIVO, semilla=config.SEMILLA):
    shutil.copyfile(db_completa, db_train)
    if os.path.exists(db_test):
        os.remove(db_test)

    azar = random.Random(semilla)
    train = sqlite3.connect(db_train)
    test = sqlite3.connect(db_test)
    test.execute("""
        CREATE TABLE interacciones (
            id_lector TEXT,
            id_libro TEXT,
            fecha TEXT,
            rating INT,
            PRIMARY KEY (id_lector, id_libro)
        )
    """)

    # Traemos todo de una vez (461.000 filas) y armamos un diccionario por lector
    por_lector = {}
    for id_lector, id_libro, fecha, rating in train.execute(
            "SELECT id_lector, id_libro, fecha, rating FROM interacciones "
            "WHERE id_libro IN (SELECT id_libro FROM libros) "
            "ORDER BY id_lector, id_libro"):
        por_lector.setdefault(id_lector, []).append((id_lector, id_libro, fecha, rating))
    totales = dict(train.execute("SELECT id_lector, COUNT(*) FROM interacciones GROUP BY id_lector"))

    reservadas_total = 0
    lectores_con_test = 0
    for id_lector in sorted(por_lector):
        total = totales[id_lector]
        positivas = [f for f in por_lector[id_lector] if f[3] is not None and f[3] >= rating_positivo]

        cantidad = max(1, round(fraccion_test * total))
        cantidad = min(cantidad, len(positivas), total - min_que_quedan)
        if cantidad <= 0:
            continue

        reservadas = azar.sample(positivas, cantidad)
        test.executemany("INSERT INTO interacciones VALUES (?, ?, ?, ?)", reservadas)
        train.executemany("DELETE FROM interacciones WHERE id_lector = ? AND id_libro = ?",
                          [(r[0], r[1]) for r in reservadas])
        reservadas_total += cantidad
        lectores_con_test += 1

    train.execute("DELETE FROM vistos")
    train.commit()
    test.commit()
    train.close()
    test.close()

    print(f"Lectores con reseñas: {len(totales)}")
    print(f"Lectores con reseñas reservadas para test: {lectores_con_test}")
    print(f"Reseñas reservadas para test: {reservadas_total}")
    print(f"Guardado en {os.path.basename(db_train)} y {os.path.basename(db_test)}")


# ===========================================================================
# Corte temporal
# ===========================================================================

def leer_fecha(texto):
    """Las fechas vienen como "dd-mm-aaaa" (y unas pocas, cargadas por la
    pagina, como "aaaa-mm-dd"). Devuelve un datetime, o None si no se entiende."""
    for formato in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(texto, formato)
        except (TypeError, ValueError):
            pass
    return None


def separar_train_test_temporal(fecha_corte, db_completa=config.DB_COMPLETA, db_train=config.DB_TRAIN,
                                db_test=config.DB_TEST, min_en_train=config.MIN_QUE_QUEDAN,
                                rating_positivo=config.RATING_POSITIVO):
    """Corte global por fecha: el modelo aprende con el pasado y se lo evalua
    con el futuro, como en la realidad (y probablemente como en Kaggle).

    - train: todas las lecturas ANTERIORES a fecha_corte ("aaaa-mm-dd").
    - test : las lecturas positivas desde fecha_corte en adelante, solo de
             lectores con al menos min_en_train lecturas antes del corte
             (a un lector sin pasado no hay como evaluarlo).
    - Todas las lecturas posteriores al corte salen de train, aunque no vayan
      a test: si no, el modelo veria el futuro de otros lectores.
    - Las pocas lecturas sin fecha legible se quedan en train.
    """
    corte = datetime.strptime(fecha_corte, "%Y-%m-%d")
    shutil.copyfile(db_completa, db_train)
    if os.path.exists(db_test):
        os.remove(db_test)

    train = sqlite3.connect(db_train)
    test = sqlite3.connect(db_test)
    test.execute("""
        CREATE TABLE interacciones (
            id_lector TEXT,
            id_libro TEXT,
            fecha TEXT,
            rating INT,
            PRIMARY KEY (id_lector, id_libro)
        )
    """)

    futuras = []
    lecturas_antes = {}
    for fila in train.execute("SELECT id_lector, id_libro, fecha, rating FROM interacciones"):
        fecha = leer_fecha(fila[2])
        if fecha is not None and fecha >= corte:
            futuras.append(fila)
        else:
            lecturas_antes[fila[0]] = lecturas_antes.get(fila[0], 0) + 1

    en_catalogo = {b for (b,) in train.execute("SELECT id_libro FROM libros")}
    para_test = [f for f in futuras
                 if lecturas_antes.get(f[0], 0) >= min_en_train
                 and f[3] is not None and f[3] >= rating_positivo
                 and f[1] in en_catalogo]

    test.executemany("INSERT INTO interacciones VALUES (?, ?, ?, ?)", para_test)
    train.executemany("DELETE FROM interacciones WHERE id_lector = ? AND id_libro = ?",
                      [(f[0], f[1]) for f in futuras])
    train.execute("DELETE FROM vistos")
    train.commit()
    test.commit()
    train.close()
    test.close()

    print(f"Corte: {fecha_corte}")
    print(f"Lecturas posteriores al corte (salen de train): {len(futuras)}")
    print(f"Lecturas en test (positivas, de lectores con pasado): {len(para_test)}")
    print(f"Lectores en test: {len({f[0] for f in para_test})}")
    print(f"Guardado en {os.path.basename(db_train)} y {os.path.basename(db_test)}")


def separar_train_test_ultimas(db_completa=config.DB_COMPLETA, db_train=config.DB_TRAIN, db_test=config.DB_TEST,
                               fraccion_test=config.FRACCION_TEST, min_que_quedan=config.MIN_QUE_QUEDAN,
                               rating_positivo=config.RATING_POSITIVO, cantidad_fija=None):
    """Corte temporal POR LECTOR: a cada lector se le esconde su ultimo
    'fraccion_test' de lecturas, ordenadas por fecha. Con cantidad_fija=20
    se le esconden sus ultimas 20 lecturas (o menos, si no le quedarian
    min_que_quedan en train).

    - Esas ultimas lecturas salen todas de train (tambien las de rating bajo),
      para que el modelo no vea el "futuro" de ese lector.
    - A test van solo las positivas (rating >= rating_positivo).
    - Cada lector conserva al menos min_que_quedan lecturas en train.
    - Las lecturas con fecha ilegible se tratan como las mas viejas (quedan en train).
    - Diferencia con el corte global: las ultimas lecturas de un lector pueden ser
      de 2012 y las de otro de 2024, asi que train puede incluir lecturas de otros
      lectores posteriores a las de test.
    """
    shutil.copyfile(db_completa, db_train)
    if os.path.exists(db_test):
        os.remove(db_test)

    train = sqlite3.connect(db_train)
    test = sqlite3.connect(db_test)
    test.execute("""
        CREATE TABLE interacciones (
            id_lector TEXT,
            id_libro TEXT,
            fecha TEXT,
            rating INT,
            PRIMARY KEY (id_lector, id_libro)
        )
    """)

    en_catalogo = {b for (b,) in train.execute("SELECT id_libro FROM libros")}
    por_lector = {}
    for fila in train.execute("SELECT id_lector, id_libro, fecha, rating FROM interacciones"):
        por_lector.setdefault(fila[0], []).append(fila)

    muy_vieja = datetime(1900, 1, 1)
    a_sacar, para_test = [], []
    for id_lector in sorted(por_lector):
        filas = por_lector[id_lector]
        # orden por fecha (y por id_libro para desempatar el mismo dia, asi es reproducible)
        filas.sort(key=lambda f: (leer_fecha(f[2]) or muy_vieja, f[1]))
        if cantidad_fija:
            cantidad = min(cantidad_fija, len(filas) - min_que_quedan)
        else:
            cantidad = min(max(1, round(fraccion_test * len(filas))), len(filas) - min_que_quedan)
        if cantidad <= 0:
            continue
        ultimas = filas[-cantidad:]
        a_sacar.extend(ultimas)
        para_test.extend(f for f in ultimas
                         if f[3] is not None and f[3] >= rating_positivo and f[1] in en_catalogo)

    test.executemany("INSERT INTO interacciones VALUES (?, ?, ?, ?)", para_test)
    train.executemany("DELETE FROM interacciones WHERE id_lector = ? AND id_libro = ?",
                      [(f[0], f[1]) for f in a_sacar])
    train.execute("DELETE FROM vistos")
    train.commit()
    test.commit()
    train.close()
    test.close()

    criterio = f"las ultimas {cantidad_fija}" if cantidad_fija else f"el ultimo {fraccion_test:.0%}"
    print(f"Lecturas sacadas de train ({criterio} de cada lector): {len(a_sacar)}")
    print(f"Lecturas en test (las positivas de esas): {len(para_test)}")
    print(f"Lectores en test: {len({f[0] for f in para_test})}")
    print(f"Guardado en {os.path.basename(db_train)} y {os.path.basename(db_test)}")

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

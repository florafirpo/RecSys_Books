"""Tests anti-leakage de la separacion train / test."""

import hashlib
import sqlite3

import pytest

from conftest import crear_base
from dataset import separar_train_test


def pares(ruta):
    con = sqlite3.connect(ruta)
    resultado = {(u, b) for u, b in con.execute("SELECT id_lector, id_libro FROM interacciones")}
    con.close()
    return resultado


@pytest.fixture
def split(base, tmp_path):
    db_train, db_test = str(tmp_path / "train.db"), str(tmp_path / "test.db")
    separar_train_test(base, db_train, db_test, fraccion_test=0.2, min_que_quedan=3, rating_positivo=8)
    return base, db_train, db_test


def test_ninguna_reseña_de_test_queda_en_train(split):
    _, db_train, db_test = split
    assert pares(db_train) & pares(db_test) == set()


def test_train_mas_test_es_la_base_original(split):
    db, db_train, db_test = split
    assert pares(db_train) | pares(db_test) == pares(db)


def test_test_solo_tiene_reseñas_positivas(split):
    _, _, db_test = split
    con = sqlite3.connect(db_test)
    ratings = [r for (r,) in con.execute("SELECT rating FROM interacciones")]
    assert ratings and min(ratings) >= 8


def test_cada_lector_conserva_el_minimo_en_train(split):
    _, db_train, db_test = split
    en_train = [u for u, _ in pares(db_train)]
    assert all(en_train.count(u) >= 3 for u, _ in pares(db_test))
    assert "dani" not in {u for u, _ in pares(db_test)}  # solo tiene 2 reseñas


def test_se_reserva_la_fraccion_pedida(tmp_path):
    libros = [(f"L{k}", "Narrativa") for k in range(50)]
    interacciones = [("ana", f"L{k}", 9) for k in range(50)]
    db = crear_base(tmp_path / "data.db", libros, interacciones)
    separar_train_test(db, str(tmp_path / "train.db"), str(tmp_path / "test.db"), fraccion_test=0.2)
    assert len(pares(str(tmp_path / "test.db"))) == 10


def test_la_separacion_es_reproducible(base, tmp_path):
    for carpeta in ("a", "b"):
        (tmp_path / carpeta).mkdir()
        separar_train_test(base, str(tmp_path / carpeta / "train.db"), str(tmp_path / carpeta / "test.db"))
    assert pares(str(tmp_path / "a" / "test.db")) == pares(str(tmp_path / "b" / "test.db"))


def test_la_base_original_no_se_modifica(base, tmp_path):
    with open(base, "rb") as f:
        antes = hashlib.sha256(f.read()).hexdigest()
    separar_train_test(base, str(tmp_path / "train.db"), str(tmp_path / "test.db"))
    with open(base, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == antes


# ---------------------------------------------------------------------------
# Corte temporal
# ---------------------------------------------------------------------------

from dataset import leer_fecha, separar_train_test_temporal  # noqa: E402


def crear_base_con_fechas(tmp_path):
    """ana y beto leen antes y despues del corte; caro solo despues."""
    db = crear_base(tmp_path / "data.db")
    con = sqlite3.connect(db)
    con.execute("DELETE FROM interacciones")
    con.executemany("INSERT INTO interacciones VALUES (?, ?, ?, ?)", [
        ("ana", "L1", "10-05-2022", 9), ("ana", "L2", "11-05-2022", 8), ("ana", "L3", "12-05-2022", 7),
        ("ana", "L4", "03-02-2024", 9),   # futura positiva -> test
        ("ana", "L5", "04-02-2024", 4),   # futura negativa -> sale de train, no va a test
        ("beto", "L1", "01-01-2021", 9), ("beto", "L2", "02-01-2021", 9), ("beto", "L4", "03-01-2021", 9),
        ("beto", "L6", "2026-09-19", 10),  # formato de la pagina web, tambien futura
        ("caro", "L1", "05-06-2024", 10),  # sin pasado: no se puede evaluar
        ("dani", "L2", "muy poco creibles", 8),  # fecha ilegible: queda en train
    ])
    con.commit()
    con.close()
    db_train, db_test = str(tmp_path / "train.db"), str(tmp_path / "test.db")
    separar_train_test_temporal("2024-01-01", db, db_train, db_test, min_en_train=3, rating_positivo=8)
    return db, db_train, db_test


def test_leer_fecha_entiende_los_dos_formatos():
    assert leer_fecha("03-02-2024").year == 2024
    assert leer_fecha("2026-09-19").month == 9
    assert leer_fecha("muy poco creibles") is None


def test_temporal_train_no_tiene_nada_posterior_al_corte(tmp_path):
    _, db_train, _ = crear_base_con_fechas(tmp_path)
    con = sqlite3.connect(db_train)
    fechas = [leer_fecha(f) for (f,) in con.execute("SELECT fecha FROM interacciones")]
    assert all(f is None or f.year < 2024 for f in fechas)


def test_temporal_test_solo_positivas_de_lectores_con_pasado(tmp_path):
    _, _, db_test = crear_base_con_fechas(tmp_path)
    assert pares(db_test) == {("ana", "L4"), ("beto", "L6")}


def test_temporal_fecha_ilegible_queda_en_train(tmp_path):
    _, db_train, _ = crear_base_con_fechas(tmp_path)
    assert ("dani", "L2") in pares(db_train)


# ---------------------------------------------------------------------------
# Ultimo 20% por lector
# ---------------------------------------------------------------------------

from dataset import separar_train_test_ultimas  # noqa: E402


def test_ultimas_esconde_las_mas_recientes_de_cada_lector(tmp_path):
    # ana tiene 10 lecturas, una por año; con 20% se le esconden las 2 ultimas (2028 y 2029)
    libros = [(f"L{k}", "Narrativa") for k in range(10)]
    interacciones = [("ana", f"L{k}", 9) for k in range(10)]
    db = crear_base(tmp_path / "data.db", libros, interacciones)
    con = sqlite3.connect(db)
    for k in range(10):
        con.execute("UPDATE interacciones SET fecha = ? WHERE id_libro = ?", (f"01-01-{2020 + k}", f"L{k}"))
    con.execute("UPDATE interacciones SET rating = 3 WHERE id_libro = 'L8'")  # una de las ultimas es negativa
    con.commit()
    con.close()

    db_train, db_test = str(tmp_path / "train.db"), str(tmp_path / "test.db")
    separar_train_test_ultimas(db, db_train, db_test, fraccion_test=0.2, min_que_quedan=3, rating_positivo=8)
    assert pares(db_test) == {("ana", "L9")}  # solo la positiva va a test
    assert ("ana", "L8") not in pares(db_train)  # pero la negativa tambien sale de train
    assert len(pares(db_train)) == 8


def test_ultimos_n_esconde_una_cantidad_fija(tmp_path):
    # ana tiene 30 lecturas positivas: con N=20 se le esconden 20 y quedan 10 en train;
    # beto tiene 5: solo se le pueden esconder 2 (le quedan 3 en train)
    libros = [(f"L{k}", "Narrativa") for k in range(30)]
    interacciones = [("ana", f"L{k}", 9) for k in range(30)] + [("beto", f"L{k}", 9) for k in range(5)]
    db = crear_base(tmp_path / "data.db", libros, interacciones)
    db_train, db_test = str(tmp_path / "train.db"), str(tmp_path / "test.db")
    separar_train_test_ultimas(db, db_train, db_test, min_que_quedan=3, rating_positivo=8, cantidad_fija=20)
    en_test = [u for u, _ in pares(db_test)]
    assert en_test.count("ana") == 20 and en_test.count("beto") == 2
    assert pares(db_train) & pares(db_test) == set()

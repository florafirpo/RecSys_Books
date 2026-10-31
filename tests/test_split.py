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

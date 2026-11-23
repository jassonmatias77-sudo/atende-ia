"""Testes da configuração: validação explícita."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Config, ConfigAuth, ConfigBase, ConfigFila


def test_defaults_validos():
    c = Config()
    assert c.base.prefixo == "atende_"
    assert c.base.dimensoes == 768
    assert c.fila.backend in ("memoria", "redis")


def test_limiar_calibrado_por_medicao():
    """0.63 veio de medição: separa relevantes (0.648+) de irrelevantes (0.604)."""
    assert abs(Config().limiar_escalonamento - 0.63) < 1e-9


def test_senha_da_base_obrigatoria():
    with pytest.raises(ValidationError) as e:
        ConfigBase(password="")
    assert "PG_PASSWORD" in str(e.value)


def test_segredo_jwt_minimo():
    with pytest.raises(ValidationError) as e:
        ConfigAuth(secret="curto")
    assert "16 caracteres" in str(e.value)


def test_backend_de_fila_invalido_recusado():
    with pytest.raises(ValidationError):
        ConfigFila(backend="kafka")


def test_dimensoes_dentro_do_limite():
    with pytest.raises(ValidationError):
        ConfigBase(password="x" * 10, dimensoes=10)


def test_dsn_nao_expoe_senha_em_repr():
    """O repr do modelo não pode imprimir a senha."""
    b = ConfigBase(password="segredo_muito_secreto")
    assert "segredo_muito_secreto" not in repr(b)


def test_dsn_montado_corretamente():
    b = ConfigBase(password="p", host="h", port=1234, database="d", user="u")
    assert "host=h" in b.dsn() and "port=1234" in b.dsn()
    assert "dbname=postgres" in b.dsn_sem_base()

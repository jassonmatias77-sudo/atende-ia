"""Testes de autenticação: hash, JWT e recusas."""
from __future__ import annotations

import time

import jwt
import pytest

from app.auth import ErroAuth, gerar_hash, verificar_senha
from app.config import config


# --------------------------------------------------------------------- hash
def test_hash_nao_guarda_a_senha():
    h = gerar_hash("minha_senha_123")
    assert "minha_senha_123" not in h
    assert h.startswith("$2")            # prefixo do bcrypt


def test_hash_e_salt_unico():
    """Duas chamadas com a mesma senha dão hashes diferentes (salt)."""
    a, b = gerar_hash("senha_igual"), gerar_hash("senha_igual")
    assert a != b
    assert verificar_senha("senha_igual", a)
    assert verificar_senha("senha_igual", b)


def test_verificar_senha_errada():
    h = gerar_hash("senha_certa")
    assert not verificar_senha("senha_errada", h)


def test_hash_recusa_senha_curta():
    with pytest.raises(ErroAuth):
        gerar_hash("abc")


def test_verificar_com_hash_corrompido_nao_explode():
    assert verificar_senha("x", "isto-nao-e-um-hash") is False


# ---------------------------------------------------------------------- JWT
class _BD:
    """Duplo mínimo: só o cursor é usado por ServicoAuth."""

    def __init__(self, linha=None):
        self._linha = linha

    def cursor(self):
        bd = self

        class _Cur:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def execute(self_, *a, **k):
                pass

            def fetchone(self_):
                return bd._linha

        return _Cur()


from app.auth import ServicoAuth  # noqa: E402


def test_token_valido_carrega_claims():
    s = ServicoAuth(bd=_BD())
    t = s._emitir(7, "a@b.com", "admin")
    d = s.validar(t)
    assert d["sub"] == "7"
    assert d["email"] == "a@b.com"
    assert d["papel"] == "admin"
    assert "exp" in d


def test_token_expirado_recusado(monkeypatch):
    s = ServicoAuth(bd=_BD())
    cfg = config()
    passado = int(time.time()) - 120
    t = jwt.encode({"sub": "1", "exp": passado}, cfg.auth.secret,
                   algorithm=cfg.auth.algoritmo)
    with pytest.raises(ErroAuth) as e:
        s.validar(t)
    assert "expirado" in str(e.value)


def test_token_assinatura_errada_recusado():
    cfg = config()
    t = jwt.encode({"sub": "1", "exp": int(time.time()) + 60},
                   "outro_segredo_completamente_diferente",
                   algorithm=cfg.auth.algoritmo)
    with pytest.raises(ErroAuth):
        ServicoAuth(bd=_BD()).validar(t)


def test_token_lixo_recusado():
    with pytest.raises(ErroAuth):
        ServicoAuth(bd=_BD()).validar("nao.e.um.token")


def test_autenticar_utilizador_inexistente():
    """E-mail e senha errada dão a MESMA mensagem (não revela e-mails)."""
    s = ServicoAuth(bd=_BD(None))
    with pytest.raises(ErroAuth) as e:
        s.autenticar("naoexiste@x.com", "senha123")
    assert "credenciais" in str(e.value)


def test_autenticar_senha_errada_mesma_mensagem():
    h = gerar_hash("certa_senha")
    s = ServicoAuth(bd=_BD({"id": 1, "email": "a@b.com",
                            "senha_hash": h, "papel": "atendente"}))
    with pytest.raises(ErroAuth) as e:
        s.autenticar("a@b.com", "errada")
    assert "credenciais" in str(e.value)


def test_autenticar_com_sucesso_devolve_token():
    h = gerar_hash("certa_senha")
    s = ServicoAuth(bd=_BD({"id": 1, "email": "a@b.com",
                            "senha_hash": h, "papel": "admin"}))
    t = s.autenticar("a@b.com", "certa_senha")
    assert t
    d = s.validar(t)
    assert d["papel"] == "admin"

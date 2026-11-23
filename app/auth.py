"""Autenticação JWT e hash de senhas.

DECISÕES QUE IMPORTAM

1. BCRYPT, NUNCA SHA/MD5
    Um hash rápido é um hash quebrado. O bcrypt é deliberadamente lento:
    ~100 ms por verificação. Para o utilizador é imperceptível; para quem
    tenta força bruta, é 100 mil vezes mais caro.

2. O TOKEN NÃO GUARDA A SENHA
    Guarda apenas o e-mail (claim `sub`) e o papel. Se o token vazar, o
    atacante tem acesso temporário — não as credenciais.

3. EXPIRAÇÃO SEMPRE
    Token sem expiração é uma porta aberta permanente. 60 minutos por
    omissão, configurável.

4. A SENHA NUNCA APARECE EM LOG
    Nem em erro, nem em debug. O código evita construir mensagens que a
    incluam.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Config, config as cfg_global
from app.database import BaseDados

log = logging.getLogger(__name__)

esquema = HTTPBearer(auto_error=False)


class ErroAuth(RuntimeError):
    """Credenciais inválidas ou token expirado."""


def gerar_hash(senha: str) -> str:
    """Hash bcrypt com custo padrão (12 rounds)."""
    if not senha or len(senha) < 6:
        raise ErroAuth("senha tem de ter pelo menos 6 caracteres")
    return bcrypt.hashpw(senha.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verificar_senha(senha: str, hash_guardado: str) -> bool:
    """Comparação em tempo constante — evita timing attack."""
    try:
        return bcrypt.checkpw(senha.encode("utf-8"),
                              hash_guardado.encode("utf-8"))
    except Exception:  # noqa: BLE001
        return False


class ServicoAuth:
    """Emissão e validação de tokens."""

    def __init__(self, cfg: Config | None = None,
                 bd: BaseDados | None = None) -> None:
        self.cfg = cfg or cfg_global()
        self.bd = bd or BaseDados(self.cfg)

    # ----------------------------------------------------------- utilizadores
    def criar_utilizador(self, email: str, senha: str,
                         papel: str = "atendente") -> int:
        """Cria um utilizador. Devolve o id."""
        hash_ = gerar_hash(senha)
        with self.bd.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.cfg.base.prefixo}usuarios "
                f"(email, senha_hash, papel) VALUES (%s, %s, %s) "
                f"ON CONFLICT (email) DO UPDATE "
                f"SET senha_hash = EXCLUDED.senha_hash, papel = EXCLUDED.papel "
                f"RETURNING id",
                (email.lower().strip(), hash_, papel))
            return int(cur.fetchone()["id"])

    def autenticar(self, email: str, senha: str) -> str:
        """Devolve o token, ou levanta ErroAuth.

        A mesma mensagem para e-mail inexistente e senha errada: dizer qual
        falhou permite a quem ataca descobrir e-mails válidos.
        """
        with self.bd.cursor() as cur:
            cur.execute(
                f"SELECT id, email, senha_hash, papel "
                f"FROM {self.cfg.base.prefixo}usuarios WHERE email = %s",
                (email.lower().strip(),))
            u = cur.fetchone()

        if not u or not verificar_senha(senha, u["senha_hash"]):
            raise ErroAuth("credenciais inválidas")

        return self._emitir(int(u["id"]), u["email"], u["papel"])

    def _emitir(self, uid: int, email: str, papel: str) -> str:
        agora = datetime.now(timezone.utc)
        payload = {
            "sub": str(uid),
            "email": email,
            "papel": papel,
            "iat": int(agora.timestamp()),
            "exp": int((agora + timedelta(minutes=self.cfg.auth.expira_min))
                       .timestamp()),
        }
        return jwt.encode(payload, self.cfg.auth.secret,
                          algorithm=self.cfg.auth.algoritmo)

    def validar(self, token: str) -> dict:
        """Valida o token. Levanta ErroAuth com motivo claro."""
        try:
            return jwt.decode(token, self.cfg.auth.secret,
                              algorithms=[self.cfg.auth.algoritmo])
        except jwt.ExpiredSignatureError:
            raise ErroAuth("token expirado") from None
        except jwt.InvalidTokenError as e:
            raise ErroAuth("token inválido (%s)" % type(e).__name__) from e


# ---------------------------------------------------------------------------
# Dependências do FastAPI
# ---------------------------------------------------------------------------
def utilizador_actual(credenciais: HTTPAuthorizationCredentials | None = Depends(esquema)):
    """Dependência: exige um token válido."""
    if credenciais is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token ausente. Envie 'Authorization: Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"})
    try:
        return ServicoAuth().validar(credenciais.credentials)
    except ErroAuth as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}) from e


def exigir_admin(u: dict = Depends(utilizador_actual)) -> dict:
    """Dependência: exige papel 'admin'."""
    if u.get("papel") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esta operação exige papel de administrador.")
    return u

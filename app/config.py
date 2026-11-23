"""Configuração da plataforma, validada por Pydantic.

POR QUE CONFIGURAÇÃO TIPADA
    Num sistema com 6 serviços (API, base, fila, LLM, métricas, frontend),
    um erro de configuração descoberto em produção custa horas. Com Pydantic,
    o mesmo erro mata o arranque com mensagem explícita.

    Além disso: segredos nunca no código. Tudo vem de variáveis de ambiente
    ou de um .env não versionado.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


def _env(chave: str, padrao: str = "") -> str:
    return os.environ.get(chave, padrao) or padrao


class ConfigBase(BaseModel):
    """Parâmetros da base de dados PostgreSQL com pgvector."""

    host: str = Field(default_factory=lambda: _env("PG_HOST", "192.168.0.6"))
    port: int = Field(default_factory=lambda: int(_env("PG_PORT", "5433")))
    user: str = Field(default_factory=lambda: _env("PG_USER", "jaspem"))
    # repr=False: um print(config) num log de arranque NAO pode expor
    # a senha da base. Sem isto, o Pydantic imprime o valor.
    password: str = Field(default_factory=lambda: _env("PG_PASSWORD", ""),
                          repr=False)
    database: str = Field(default_factory=lambda: _env("PG_DATABASE", "jaspem_vector"))

    # prefixo das tabelas deste projecto — evita colidir com as do JASPEM
    prefixo: str = Field(default="atende_")

    # pgvector: 768 dimensões com nomic-embed-text
    dimensoes: int = Field(default=768, ge=64, le=4096)
    pool_min: int = Field(default=1, ge=1)
    pool_max: int = Field(default=5, ge=1)

    @field_validator("password")
    @classmethod
    def senha_obrigatoria(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "PG_PASSWORD vazio. Defina a variável de ambiente ou use "
                "o .env (ver .env.example).")
        return v

    def dsn(self) -> str:
        """String de ligação para psycopg2."""
        return ("host=%s port=%d dbname=%s user=%s password=%s "
                "connect_timeout=10" % (self.host, self.port, self.database,
                                        self.user, self.password))

    def dsn_sem_base(self) -> str:
        """Ligação à base 'postgres' — para criar a nossa."""
        return ("host=%s port=%d dbname=postgres user=%s password=%s "
                "connect_timeout=10" % (self.host, self.port, self.user,
                                        self.password))


class ConfigLLM(BaseModel):
    """Modelo de linguagem local (Ollama)."""

    base_url: str = Field(default_factory=lambda: _env(
        "OLLAMA_URL", "http://127.0.0.1:11434"))
    modelo: str = Field(default_factory=lambda: _env(
        "ATENDE_LLM", "qwen2.5-coder:7b"))
    modelo_embedding: str = Field(default_factory=lambda: _env(
        "ATENDE_EMBED", "nomic-embed-text:latest"))
    temperatura: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=8192)
    timeout_s: float = Field(default=180.0, gt=0)
    keep_alive: str = Field(default="30m")


class ConfigAuth(BaseModel):
    """Autenticação JWT."""

    secret: str = Field(default_factory=lambda: _env("ATENDE_JWT_SECRET", ""))
    algoritmo: str = Field(default="HS256")
    expira_min: int = Field(default=60, ge=1, le=10080)

    @field_validator("secret")
    @classmethod
    def segredo_minimo(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError(
                "ATENDE_JWT_SECRET tem de ter pelo menos 16 caracteres. "
                "Gere com: python -c \"import secrets;print(secrets.token_urlsafe(32))\"")
        return v


class ConfigFila(BaseModel):
    """Fila de tarefas longas.

    Redis não está disponível neste ambiente (medido: portas 6379 fechadas
    local e no OMV). O backend em memória com threads serve para demonstração
    e o README declara a limitação.
    """

    backend: str = Field(default_factory=lambda: _env("ATENDE_FILA", "memoria"))
    workers: int = Field(default=2, ge=1, le=16)
    redis_url: str = Field(default_factory=lambda: _env(
        "REDIS_URL", "redis://127.0.0.1:6379/0"))

    @field_validator("backend")
    @classmethod
    def backend_valido(cls, v: str) -> str:
        if v not in ("memoria", "redis"):
            raise ValueError("backend tem de ser 'memoria' ou 'redis'")
        return v


class Config(BaseModel):
    """Configuração completa."""

    base: ConfigBase = Field(default_factory=ConfigBase)
    llm: ConfigLLM = Field(default_factory=ConfigLLM)
    auth: ConfigAuth = Field(default_factory=ConfigAuth)
    fila: ConfigFila = Field(default_factory=ConfigFila)

    ambiente: str = Field(default_factory=lambda: _env("ATENDE_AMBIENTE", "dev"))
    max_chars_resposta: int = Field(default=1500, ge=100)
    # Limiar calibrado por MEDICAO, nao por suposicao.
    # Medido com pgvector + nomic-embed-text (22/11/2026):
    #   relevantes   -> 0.771 (troca) | 0.689 (entrega) | 0.648 (pagamento)
    #   irrelevantes -> 0.604 (outro tema) | 0.508 (tema distante)
    # A fronteira esta' entre 0.604 e 0.648; 0.63 separa os dois casos.
    limiar_escalonamento: float = Field(
        default_factory=lambda: float(
            _env("ATENDE_LIMIAR_ESCALONAMENTO", "0.63")),
        ge=0.0, le=1.0,
        description=("Abaixo deste score de similaridade, o agente escala para "
                     "humano em vez de responder. Calibrado por medicao."))


def _carregar_env() -> None:
    """Carrega o .env, se existir. Não sobrepõe variáveis já definidas."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for linha in env.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        chave, valor = chave.strip(), valor.strip().strip('"').strip("'")
        os.environ.setdefault(chave, valor)


@lru_cache(maxsize=1)
def config() -> Config:
    """Configuração única do processo, carregada uma vez."""
    _carregar_env()
    return Config()

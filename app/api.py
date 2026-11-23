"""API HTTP da plataforma de atendimento.

DECISÕES DE API

1. /health NÃO EXIGE TOKEN
    Um monitor de infraestrutura (Prometheus, load balancer) não tem
    credencial de utilizador. O /health expõe estado, nunca dados.

2. /metrics IDEM
    Segue a convenção do Prometheus: scrape anónimo dentro da rede.

3. 202 ACCEPTED para indexação
    Indexar é longo. Devolver 200 significaria "já está feito", o que é
    falso. 202 com id de tarefa é a resposta honesta.

4. ERRO DE DEPENDÊNCIA É 503, NÃO 500
    500 = "o meu código tem um bug".
    503 = "uma dependência externa está em baixo, tente depois".
    Distinguir permite ao cliente decidir se retenta.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.agente import AgenteAtendimento
from app.auth import (ErroAuth, ServicoAuth, exigir_admin,
                      utilizador_actual)
from app.config import config
from app.database import ErroBaseDados
from app.fila import criar_fila
from app.observabilidade import (REQUISICOES, actualizar_documentos,
                                 exportar, medir_etapa, registar_atendimento,
                                 registar_falha)
from app.rag import MotorRAG

log = logging.getLogger(__name__)

_cfg = config()
_agente: AgenteAtendimento | None = None
_fila = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Arranque e encerramento. Garante o schema e as dependências."""
    global _agente, _fila

    log.info("a arrancar: base=%s:%s", _cfg.base.host, _cfg.base.port)
    _agente = AgenteAtendimento(_cfg)
    _fila = criar_fila(_cfg)

    try:
        feito = _agente.bd.criar_schema()
        log.info("schema pronto: %s", feito)
        actualizar_documentos(_agente.bd.contar_documentos())
    except ErroBaseDados as e:
        # Não mata o arranque: /health dirá que está degradado.
        log.error("schema falhou: %s", e)

    yield

    if _fila is not None:
        _fila.parar()
    log.info("a encerrar")


app = FastAPI(
    title="Atende-IA",
    description=(
        "Plataforma de atendimento ao cliente com IA. Responde com base na "
        "documentação interna e **escala para humano** quando não tem "
        "confiança suficiente."),
    version="1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def _metricas(request: Request, call_next):
    t0 = time.perf_counter()
    resposta = await call_next(request)
    dur = time.perf_counter() - t0
    rota = request.url.path
    # não contamina as métricas com /metrics e /health
    if rota not in ("/metrics", "/health"):
        REQUISICOES.labels(metodo=request.method, rota=rota,
                           estado=str(resposta.status_code)).inc()
        from app.observabilidade import LATENCIA
        LATENCIA.labels(etapa="http_total").observe(dur)
    return resposta


# --------------------------------------------------------------------- modelos
class LoginPedido(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    senha: str = Field(min_length=6, max_length=200)


class TokenResposta(BaseModel):
    token: str
    tipo: str = "bearer"
    expira_min: int


class PerguntaPedido(BaseModel):
    pergunta: str = Field(min_length=1, max_length=2000)
    sessao: str = Field(default="", max_length=100)


class RespostaAtendimento(BaseModel):
    resposta: str
    confianca: float
    escalado: bool
    fontes: list[str]
    categoria: str
    motivo: str
    tempo_s: float
    tokens: int


class IndexarPedido(BaseModel):
    pasta: str = Field(default="docs", max_length=500)
    extensoes: list[str] = Field(default_factory=lambda: [".md", ".txt"])


class TarefaResposta(BaseModel):
    tarefa_id: str
    estado: str
    mensagem: str


# ----------------------------------------------------------------------- saúde
@app.get("/health")
def health() -> dict:
    """Estado do sistema. Sem token — é para monitores."""
    global _agente
    if _agente is None:
        return {"status": "arrancando"}

    ok, msg = _agente.bd.saudavel()
    llm = _agente.llm.diagnostico()

    problemas = []
    if not ok:
        problemas.append("base de dados")
    if not llm["ollama"]:
        problemas.append("ollama")
    problemas.extend(llm.get("faltando", []))

    # degradado: a base funciona mas falta o LLM (ou vice-versa)
    estado = "ok" if not problemas else (
        "degradado" if (ok or llm["ollama"]) else "fora")

    return {
        "status": estado,
        "base": {"ok": ok, "msg": msg},
        "ollama": llm["ollama"],
        "documentos": _agente.bd.contar_documentos() if ok else 0,
        "limiar_escalonamento": _cfg.limiar_escalonamento,
        "problemas": problemas,
    }


@app.get("/metrics")
def metrics():
    """Métricas Prometheus (formato texto)."""
    payload, content_type = exportar()
    return Response(content=payload, media_type=content_type)


# ----------------------------------------------------------------------- auth
@app.post("/auth/token", response_model=TokenResposta)
def login(p: LoginPedido) -> TokenResposta:
    """Troca credenciais por um token JWT."""
    try:
        token = ServicoAuth(_cfg).autenticar(p.email, p.senha)
    except ErroAuth as e:
        # 401 com a mesma mensagem para e-mail e senha errados
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=str(e)) from e
    except ErroBaseDados as e:
        registar_falha("postgres")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="base de dados indisponível") from e
    return TokenResposta(token=token, expira_min=_cfg.auth.expira_min)


@app.post("/auth/utilizador")
def criar_utilizador(p: LoginPedido,
                     _admin: dict = Depends(exigir_admin)) -> dict:
    """Cria ou actualiza um utilizador. Só admin."""
    try:
        uid = ServicoAuth(_cfg).criar_utilizador(p.email, p.senha)
    except ErroAuth as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"id": uid, "email": p.email.lower().strip()}


@app.get("/auth/eu")
def quem_sou(u: dict = Depends(utilizador_actual)) -> dict:
    """Devolve os dados do token actual."""
    return {"id": u.get("sub"), "email": u.get("email"),
            "papel": u.get("papel")}


# ------------------------------------------------------------------ atendimento
@app.post("/perguntar", response_model=RespostaAtendimento)
def perguntar(p: PerguntaPedido,
              u: dict = Depends(utilizador_actual)) -> RespostaAtendimento:
    """Atende uma pergunta do cliente."""
    if _agente is None:
        raise HTTPException(status_code=503, detail="a arrancar")

    sessao = p.sessao or ("u" + str(u.get("sub", "anon")))

    with medir_etapa("agente_total"):
        r = _agente.atender(p.pergunta, sessao)

    registar_atendimento(r.categoria, r.escalado, r.confianca, r.motivo)

    if r.erro and not r.resposta:
        dependencia = "ollama" if "LLM" in r.erro or "modelo" in r.erro else "postgres"
        registar_falha(dependencia)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="serviço indisponível: %s" % r.erro[:150])

    return RespostaAtendimento(
        resposta=r.resposta, confianca=round(r.confianca, 4),
        escalado=r.escalado, fontes=r.fontes, categoria=r.categoria,
        motivo=r.motivo, tempo_s=round(r.tempo_s, 3), tokens=r.tokens)


@app.get("/historico")
def historico(sessao: str = "", limite: int = 10,
              u: dict = Depends(utilizador_actual)) -> dict:
    """Histórico de uma sessão."""
    if _agente is None:
        raise HTTPException(status_code=503, detail="a arrancar")
    sid = sessao or ("u" + str(u.get("sub", "")))
    return {"sessao": sid, "conversas": _agente.bd.historico(sid, limite)}


@app.get("/metricas-negocio")
def metricas_negocio(_u: dict = Depends(utilizador_actual)) -> dict:
    """Números de negócio: confiança média, escalonamentos, tempo."""
    if _agente is None:
        raise HTTPException(status_code=503, detail="a arrancar")
    return _agente.bd.metricas()


# --------------------------------------------------------------------- indexar
@app.post("/indexar", response_model=TarefaResposta,
          status_code=status.HTTP_202_ACCEPTED)
def indexar(p: IndexarPedido,
            _u: dict = Depends(utilizador_actual)) -> TarefaResposta:
    """Enfileira a indexação de uma pasta. Devolve 202 + id da tarefa."""
    if _agente is None or _fila is None:
        raise HTTPException(status_code=503, detail="a arrancar")

    pasta = Path(p.pasta)
    if not pasta.is_absolute():
        pasta = Path(__file__).resolve().parent.parent / p.pasta

    def trabalho(pasta=str(pasta), extensoes=tuple(p.extensoes)):
        rag = MotorRAG(_cfg, _agente.bd, _agente.llm)
        resumo = rag.indexar_pasta(pasta, extensoes)
        actualizar_documentos(_agente.bd.contar_documentos())
        return resumo

    tid = _fila.submeter("indexar", trabalho)
    return TarefaResposta(tarefa_id=tid, estado="pendente",
                          mensagem="indexação em curso; consulte /tarefa/%s" % tid)


@app.get("/tarefa/{tid}")
def tarefa(tid: str, _u: dict = Depends(utilizador_actual)) -> dict:
    """Estado de uma tarefa em background."""
    if _fila is None:
        raise HTTPException(status_code=503, detail="a arrancar")
    e = _fila.estado(tid)
    if e is None:
        raise HTTPException(status_code=404, detail="tarefa não encontrada")
    return e


@app.get("/tarefas")
def tarefas(_u: dict = Depends(utilizador_actual)) -> dict:
    """Lista de tarefas e estatísticas da fila."""
    if _fila is None:
        raise HTTPException(status_code=503, detail="a arrancar")
    return {"estatisticas": _fila.estatisticas(), "tarefas": _fila.listar()}

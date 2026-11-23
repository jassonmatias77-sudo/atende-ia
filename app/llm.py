"""Cliente Ollama com telemetria e erros distintos.

POR QUE ERROS DISTINTOS IMPORTAM
    "OllamaIndisponivel" é temporário — vale retry e degradação.
    "ModeloNaoEncontrado" exige acção humana (instalar o modelo).

    Se ambos forem a mesma excepção, o sistema tenta retry eternamente num
    erro que retry não resolve. Distinguir é o que permite decidir.

POR QUE keep_alive
    Medido no portfólio anterior: sem keep_alive, a primeira chamada após
    ocioso leva ~2s só a carregar o modelo. Com "30m", o modelo fica
    residente e a latência cai 5x.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from app.config import Config, config as cfg_global


class OllamaIndisponivel(RuntimeError):
    """Servidor não respondeu. Temporário — vale retry."""


class ModeloNaoEncontrado(RuntimeError):
    """Modelo não existe. Exige instalação."""


@dataclass
class Resposta:
    """O que uma geração devolve, com o que é preciso medir."""

    texto: str
    modelo: str
    latencia_s: float
    tokens: int = 0
    tentativas: int = 1
    meta: dict = field(default_factory=dict)


class ClienteOllama:
    """Cliente mínimo e previsível."""

    def __init__(self, cfg: Config | None = None) -> None:
        c = cfg or cfg_global()
        self.base = c.llm.base_url.rstrip("/")
        self.cfg = c.llm
        self.max_tentativas = 2

    # ------------------------------------------------------------------ saúde
    def disponivel(self, timeout: float = 8.0) -> bool:
        try:
            return requests.get(self.base + "/api/tags",
                                timeout=timeout).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def modelos(self) -> list[str]:
        try:
            r = requests.get(self.base + "/api/tags", timeout=15)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            raise OllamaIndisponivel("Ollama não respondeu: %s" % e) from e

    def diagnostico(self) -> dict:
        """Estado sem levantar excepção."""
        rel: dict = {"ollama": False, "faltando": []}
        rel["ollama"] = self.disponivel()
        if not rel["ollama"]:
            rel["faltando"].append("Ollama em %s" % self.base)
            return rel
        try:
            instalados = self.modelos()
            for preciso in (self.cfg.modelo, self.cfg.modelo_embedding):
                if preciso not in instalados:
                    rel["faltando"].append("%s (ollama pull %s)"
                                           % (preciso, preciso))
        except Exception as e:  # noqa: BLE001
            rel["faltando"].append(str(e)[:80])
        return rel

    # -------------------------------------------------------------------- chat
    def gerar(self, mensagens: list[dict], sistema: str = "") -> Resposta:
        msgs = ([{"role": "system", "content": sistema}] if sistema else []) + mensagens
        payload = {
            "model": self.cfg.modelo,
            "messages": msgs,
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": self.cfg.temperatura,
                        "num_predict": self.cfg.max_tokens},
        }

        ultimo = ""
        for tentativa in range(1, self.max_tentativas + 1):
            t0 = time.perf_counter()
            try:
                r = requests.post(self.base + "/api/chat", json=payload,
                                  timeout=self.cfg.timeout_s)
            except requests.exceptions.Timeout:
                ultimo = "timeout após %.0fs" % self.cfg.timeout_s
                if tentativa < self.max_tentativas:
                    time.sleep(1.5 * tentativa)
                    continue
                raise OllamaIndisponivel(ultimo)
            except Exception as e:  # noqa: BLE001
                ultimo = "%s: %s" % (type(e).__name__, e)
                if tentativa < self.max_tentativas:
                    time.sleep(1.5 * tentativa)
                    continue
                raise OllamaIndisponivel(ultimo)

            dur = time.perf_counter() - t0

            if r.status_code == 404:
                raise ModeloNaoEncontrado(
                    "modelo '%s' não existe. Instale: ollama pull %s"
                    % (self.cfg.modelo, self.cfg.modelo))

            if r.status_code != 200:
                ultimo = "HTTP %s: %s" % (r.status_code, r.text[:120])
                if tentativa < self.max_tentativas:
                    time.sleep(1.5 * tentativa)
                    continue
                raise OllamaIndisponivel(ultimo)

            d = r.json()
            msg = d.get("message") or {}
            return Resposta(
                texto=(msg.get("content") or "").strip(),
                modelo=self.cfg.modelo,
                latencia_s=dur,
                tokens=int(d.get("eval_count") or 0),
                tentativas=tentativa,
                meta={"done_reason": d.get("done_reason", "")},
            )

        raise OllamaIndisponivel(ultimo or "falha desconhecida")

    # --------------------------------------------------------------- embeddings
    def embeddings(self, texto: str) -> list[float]:
        """Vector de embedding. 768 dimensões com nomic-embed-text."""
        try:
            r = requests.post(self.base + "/api/embeddings",
                              json={"model": self.cfg.modelo_embedding,
                                    "prompt": texto},
                              timeout=self.cfg.timeout_s)
        except Exception as e:  # noqa: BLE001
            raise OllamaIndisponivel("embeddings: %s" % e) from e

        if r.status_code == 404:
            raise ModeloNaoEncontrado(
                "modelo de embedding '%s' não existe. Instale: ollama pull %s"
                % (self.cfg.modelo_embedding, self.cfg.modelo_embedding))
        if r.status_code != 200:
            raise OllamaIndisponivel("embeddings HTTP %s" % r.status_code)

        return r.json().get("embedding") or []

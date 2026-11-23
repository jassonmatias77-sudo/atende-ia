"""Demo interactiva (Streamlit).

POR QUE UMA DEMO E NÃO SÓ A API
    Um recrutador não vai fazer `curl` com um JWT. Uma demo mostra o produto
    a funcionar em dois cliques — e mostra o comportamento que interessa:
    a resposta com fontes, e o escalonamento quando não sabe.

    O painel lateral expõe as métricas: confiança média, escalonamentos,
    tempo. É a prova de que o sistema mede o próprio desempenho.

CORRER
    streamlit run app/demo.py
"""
from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="Atende-IA", page_icon="🤖", layout="wide")


@st.cache_resource
def _dependencias():
    """Uma única instância por sessão do Streamlit."""
    from app.agente import AgenteAtendimento
    from app.config import config

    cfg = config()
    return cfg, AgenteAtendimento(cfg)


def _estado_sistema(agente) -> dict:
    ok, msg = agente.bd.saudavel()
    llm = agente.llm.diagnostico()
    return {
        "base_ok": ok,
        "base_msg": msg,
        "ollama": llm.get("ollama", False),
        "documentos": agente.bd.contar_documentos() if ok else 0,
        "problemas": llm.get("faltando", []),
    }


def main() -> None:
    st.title("Atende-IA")
    st.caption("Atendimento ao cliente com IA · responde com base na "
               "documentação e escala para humano quando não sabe.")

    try:
        cfg, agente = _dependencias()
    except Exception as e:  # noqa: BLE001
        st.error("**Não consegui arrancar.**\n\n`%s: %s`\n\n"
                 "Verifique o `.env` (ver `.env.example`)."
                 % (type(e).__name__, e))
        st.stop()

    est = _estado_sistema(agente)

    # ------------------------------------------------------------ lateral
    with st.sidebar:
        st.header("Estado")
        st.metric("Documentos indexados", est["documentos"])
        st.write("Base de dados:", "OK" if est["base_ok"] else "FORA")
        st.write("Ollama:", "OK" if est["ollama"] else "FORA")
        st.write("Limiar de escalonamento:", cfg.limiar_escalonamento)
        if est["problemas"]:
            st.warning("Falta: " + ", ".join(est["problemas"])[:120])

        st.divider()
        st.header("Métricas de negócio")
        try:
            m = agente.bd.metricas()
            st.metric("Conversas", int(m.get("total") or 0))
            st.metric("Confiança média", "%.3f" % float(m.get("confianca_media") or 0))
            st.metric("Tempo médio", "%.2fs" % float(m.get("tempo_medio") or 0))
            st.metric("Escaladas", int(m.get("escaladas") or 0))
        except Exception:  # noqa: BLE001
            st.info("sem conversas ainda")

        st.divider()
        st.header("Indexar documentação")
        pasta = st.text_input("Pasta", value="docs")
        if st.button("Indexar agora", type="secondary"):
            with st.spinner("a indexar..."):
                try:
                    from app.rag import MotorRAG

                    rag = MotorRAG(cfg, agente.bd, agente.llm)
                    resumo = rag.indexar_pasta(os.path.join(os.getcwd(), pasta))
                    st.success("%d ficheiro(s), %d chunk(s), %d falha(s)"
                               % (resumo["ficheiros"], resumo["chunks"],
                                  resumo["falhas"]))
                    st.cache_resource.clear()
                except Exception as e:  # noqa: BLE001
                    st.error("%s: %s" % (type(e).__name__, e))

    # ------------------------------------------------------------- principal
    if est["documentos"] == 0:
        st.info("Ainda não há documentação indexada. Use **Indexar agora** na "
                "barra lateral, ou copie ficheiros `.md` para a pasta `docs/`.")

    if "historico" not in st.session_state:
        st.session_state.historico = []

    pergunta = st.chat_input("Escreva a sua pergunta…")

    if pergunta:
        with st.spinner("a pensar..."):
            r = agente.atender(pergunta, "demo")

        st.session_state.historico.append({"pergunta": pergunta, "r": r})

    for item in st.session_state.historico:
        r = item["r"]
        with st.chat_message("user"):
            st.write(item["pergunta"])

        with st.chat_message("assistant"):
            if r.escalado:
                st.warning(r.resposta or "Sem resposta.")
                st.caption("**Escalado para humano** · motivo: %s" % (r.motivo or "—"))
            elif r.erro:
                st.error("Erro: %s" % r.erro)
            else:
                st.write(r.resposta)

            c1, c2, c3 = st.columns(3)
            c1.metric("Confiança", "%.3f" % r.confianca)
            c2.metric("Tempo", "%.2fs" % r.tempo_s)
            c3.metric("Categoria", r.categoria or "—")

            if r.fontes:
                with st.expander("Fontes consultadas (%d)" % len(r.fontes)):
                    for f in r.fontes:
                        st.write("•", f)

    if st.session_state.historico:
        if st.button("Limpar conversa"):
            st.session_state.historico = []
            st.rerun()


if __name__ == "__main__":
    main()

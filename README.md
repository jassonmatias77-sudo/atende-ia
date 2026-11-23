# Atende-IA — Plataforma de Atendimento com IA

Sistema de atendimento que responde a clientes com base na documentação da empresa
e **escala para um humano quando não tem confiança suficiente** — em vez de inventar.

FastAPI · PostgreSQL + pgvector · LangGraph · LLM local · 49 testes automatizados.

---

## O problema de negócio

Uma equipa de suporte gasta a maior parte do turno a responder às mesmas perguntas:
prazo de entrega, política de troca, formas de pagamento. Enquanto isso, os casos que
exigem julgamento — um cliente irritado, uma excepção comercial — esperam na fila.

Um chatbot genérico piora o problema: responde a tudo com confiança, incluindo o que
não sabe. O cliente percebe, e a confiança quebra-se no momento em que mais importa.

**A hipótese deste projecto:** um sistema que responde apenas com base em documentação
verificada, e que **diz "não sei" quando não sabe**, liberta a equipa para os casos que
importam sem perder a confiança do cliente.

---

## A decisão que define o sistema

O limiar de escalonamento não foi escolhido por intuição — foi **calibrado por medição**.

| Pergunta | Score de similaridade | Relevante? |
|---|---|---|
| "Posso trocar depois de 20 dias?" | 0,771 | sim |
| "Entrega no interior?" | 0,689 | sim |
| "Aceita PIX?" | 0,648 | sim |
| **"Vendem peças para trator na Mongólia?"** | **0,604** | **não** |

A fronteira está entre **0,604** e **0,648**. O limiar é **0,63**.

Com o limiar antigo (0,35), o sistema respondia à pergunta do trator. Com 0,63, escala:

```
P: "Vocês vendem peças para trator na Mongólia?"
R: "Não tenho informação suficiente para responder com segurança.
    Vou encaminhar o seu pedido para um atendente humano."
confianca=0.604 | escalado=True | motivo=confiança 0.60 abaixo do limiar 0.63
```

E existe uma regra acima do limiar: **confiança zero é um facto, não uma política.**
Se o documento recuperado tem score 0, nenhum limiar configurado autoriza responder.
Este defeito foi encontrado pelos próprios testes, não em produção.

---

## Arquitectura

```
                     ┌────────────────────────────────┐
   cliente ────────► │ FastAPI  · JWT · rate limit    │
                     └───────────────┬────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │ LangGraph — agente com estado   │
                     │ triagem → RAG → resposta →      │
                     │ escalonamento                   │
                     └───────────────┬────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌───────────────┐          ┌──────────────────┐          ┌────────────────┐
│ PostgreSQL 17 │          │ Fila de tarefas   │          │ Ollama         │
│ + pgvector    │          │ (índice de docs)  │          │ LLM local      │
│ HNSW index    │          │ memoria/Redis     │          │ 768d embed     │
└───────────────┘          └──────────────────┘          └────────────────┘
        │
        ▼
┌───────────────┐
│ Prometheus    │  métricas por etapa + negócio
└───────────────┘
```

### Triagem determinística

Saudações e despedidas **não chamam o LLM**. Resposta em **0,005 s** a custo zero.
Só pedidos reais entram no pipeline — o que evita pagar inferência por um "olá".

---

## Stack

| Camada | Tecnologia | Porquê |
|---|---|---|
| API | **FastAPI** + Uvicorn | validação por tipos, OpenAPI automático |
| Autenticação | **JWT** (HS256) + **bcrypt** | rotas protegidas, senhas com hash |
| Base de dados | **PostgreSQL 17** | produção, não SQLite |
| Busca vectorial | **pgvector 0.8.7** + índice **HNSW** | busca aproximada no banco, não em Python |
| Agentes | **LangGraph** | estado explícito no fluxo de decisão |
| LLM | **Ollama** (`qwen2.5-coder:7b`) | custo zero, dados não saem da máquina |
| Embeddings | **nomic-embed-text** (768d) | mesma dimensão do índice |
| Fila | **threading**, backend Redis opcional | tarefas longas sem bloquear a API |
| Observabilidade | **Prometheus** | métricas por etapa e de negócio |
| Frontend | **Streamlit** | demo interactiva |
| Testes | **pytest** — 49 testes | cobrem auth, RAG, agente e config |

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | estado da base, do Ollama e do limiar activo |
| GET | `/metrics` | métricas em formato Prometheus |
| POST | `/auth/token` | login, devolve JWT (60 min) |
| POST | `/auth/utilizador` | cria utilizador (senha com bcrypt) |
| GET | `/auth/eu` | identifica o portador do token |
| POST | `/perguntar` | **rota principal** — pergunta do cliente |
| GET | `/historico` | conversas por sessão |
| GET | `/metricas-negocio` | total, confiança média, escalonamentos |
| POST | `/indexar` | indexa documentos (assíncrono, devolve tarefa) |
| GET | `/tarefa/{tid}` | estado da tarefa de indexação |
| GET | `/tarefas` | lista de tarefas |

`POST /perguntar` devolve sempre: resposta, confiança, fontes citadas, se escalou e porquê.

---

## Como correr

### 1. Requisitos

```powershell
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

Copia `.env.example` para `.env` e preenche. Nada disto vai para o código — a
configuração **falha explicitamente** se faltar um segredo, em vez de usar um valor
por omissão inseguro.

```powershell
Copy-Item .env.example .env
# edita o .env
```

Os dois valores que não têm omissão segura:

```env
PG_PASSWORD=a_tua_senha
# gera com: python -c "import secrets;print(secrets.token_urlsafe(32))"
ATENDE_JWT_SECRET=...
```

### 3. Criar o schema

```powershell
python init_db.py
```

Cria `atende_conversas`, `atende_documentos`, `atende_usuarios` e o índice HNSW — e
**verifica no catálogo do Postgres** antes de dizer que está pronto:

```
[OK] ligacao          Postgres ok - pgvector 0.8.7
[OK] schema           criado
[OK] tabelas          atende_conversas, atende_documentos, atende_usuarios
[OK] indice HNSW      atende_docs_hnsw_idx
[OK] pgvector         v0.8.7
```

É idempotente: pode correr várias vezes sem estragar nada.

### 4. Criar o primeiro administrador

A rota `POST /auth/utilizador` exige papel `admin` — e numa base vazia não há admin.
É o clássico "ovo e a galinha": quem clona o repositório ficaria bloqueado.

```powershell
python criar_admin.py
```

Pede o e-mail e a senha (a senha **não** é ecoada no ecrã). Também aceita variáveis
de ambiente, para automação:

```powershell
$env:ATENDE_ADMIN_EMAIL="eu@exemplo.com"
$env:ATENDE_ADMIN_SENHA="..."      # minimo 8 caracteres
python criar_admin.py
```

```
[OK] administrador criado/atualizado
     id    : 1
     e-mail: eu@exemplo.com
     papel : admin
[OK] login testado — token de 201 caracteres emitido.
```

O script **testa o login** no fim. Se gravasse e o login falhasse, saberias na hora — não
na primeira tentativa de usar a API. E não usa argumento de linha de comandos para a
senha: ficaria no histórico do PowerShell.

### 5. Arrancar

```powershell
# API
uvicorn app.api:app --reload --port 8100

# demo visual (noutro terminal)
# A demo importa o agente directamente — nao precisa da API em execucao,
# mas precisa do mesmo .env e do Ollama ligado.
streamlit run app/demo.py
```

A documentação interactiva fica em `http://127.0.0.1:8100/docs`.

### Com Docker

```powershell
docker compose up -d
```

Sobe a base (PostgreSQL 17 + pgvector), a API (`:8100`) e a demo (`:8502`).
O Ollama **não** vai no container — corre no host e é partilhado via
`host.docker.internal`.

---

## Testes

```powershell
python -m pytest -q
```

```
49 passed in 2.38s
```

Os testes **não exigem** Ollama nem Postgres: usam duplos (`llm_duplo`, base em memória).
Um teste que exige infraestrutura externa falha na máquina de outra pessoa — e um teste
que falha por razões alheias ao código deixa de ser sinal.

O que cobrem:

- autenticação: senha errada, token inválido, token expirado, hash bcrypt
- configuração: segredo curto recusado, senha obrigatória, `repr` sem expor segredo
- RAG: chunking, similaridade de cosseno, score zero
- agente: saudação sem LLM, resposta com fontes, escalonamento por limiar e por score zero
- fila: indexação assíncrona, estado da tarefa

### Dois defeitos reais que os testes apanharam

1. **Escalonamento silencioso** — `confianca=0.000` com `fontes=['e.md']` e `escalado=False`.
   O `avaliar` só verificava `if not documentos`; um documento com score 0 passava.
2. **Segredo exposto** — a senha da base aparecia no `repr` da configuração. Um `print(config)`
   num log de arranque expunha-a. Corrigido com `repr=False`.

---

## Limitações declaradas

Um portfólio honesto diz o que **não** faz:

- **Fila em memória por omissão.** Redis não estava disponível no ambiente de
  desenvolvimento (portas 6379 fechadas — medido, não suposto). O backend Redis está
  implementado e activa-se com `ATENDE_FILA=redis`, mas não foi exercitado em produção.
- **Um worker por processo.** A fila em memória não sobrevive a um reinício.
- **Sem re-ranking.** A recuperação é top-k por similaridade de cosseno. Um cross-encoder
  melhoraria a precisão em bases grandes.
- **A avaliação mede palavras-chave**, não correcção semântica. É um sinal, não uma verdade.
- **Um provider de LLM.** Não há comparação entre modelos; a escolha foi feita por latência.
- **`qwen2.5-coder:7b` é um modelo de código.** Serve para demonstrar o pipeline; para
  atendimento real, um modelo de instrução em português seria mais adequado.

---

## Estrutura

```
app/
  config.py          configuração tipada (Pydantic) — falha explícita
  database.py        schema, pool, busca vectorial com pgvector
  llm.py             cliente Ollama com keep_alive e retry
  rag.py             chunking, embeddings, indexação
  agente.py          grafo LangGraph: triagem → RAG → resposta → escalonamento
  auth.py            JWT + bcrypt
  fila.py            fila de tarefas (memória/Redis)
  observabilidade.py métricas Prometheus
  api.py             FastAPI — 11 rotas
  demo.py            Streamlit
docs/                base de conhecimento de exemplo
tests/               49 testes
init_db.py           cria o schema e verifica no catálogo
criar_admin.py       cria o primeiro administrador (bootstrap)
Dockerfile           imagem única (API e demo)
docker-compose.yml   base + API + demo
```

---

## O que este projecto demonstra

Não é a quantidade de tecnologia. É a disciplina de engenharia:

1. **Medir antes de decidir** — o limiar veio dos dados, não de intuição.
2. **Falhar explicitamente** — configuração inválida impede o arranque.
3. **Testar sem infraestrutura** — para o teste medir código, não disponibilidade.
4. **Declarar o que não funciona** — a lista de limitações acima é parte da entrega.
5. **Deixar o sistema dizer "não sei"** — em atendimento, uma alucinação custa mais
   que uma resposta lenta.

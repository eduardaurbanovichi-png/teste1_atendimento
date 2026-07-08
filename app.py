"""Interface de chat em Streamlit.

Este arquivo só cuida de UI e estado da sessão. Toda comunicação com o
modelo de linguagem fica em llm_client.py — separação de camadas simples,
mas que evita misturar lógica de negócio com apresentação mesmo num
projeto pequeno.
"""

import streamlit as st
from dotenv import load_dotenv

# Carrega variáveis do .env em desenvolvimento local. No Render, as env vars
# já vêm definidas pelo painel do serviço, então este load_dotenv() é um
# no-op inofensivo em produção (não sobrescreve nada se o .env não existir).
load_dotenv()

from llm_client import (
    KNOWN_FREE_MODELS,
    LLMConfigError,
    LLMRequestError,
    chat_completion,
)

SYSTEM_PROMPT = "Você é um assistente de chat geral, direto e prestativo."

st.set_page_config(page_title="Chat Agent", page_icon="💬")
st.title("💬 Agente de Chat")
st.caption("Streamlit + OpenRouter (camada gratuita)")

# --- Estado da sessão ---------------------------------------------------
# Histórico vive só na sessão do navegador: some ao fechar/atualizar a aba.
# Suficiente para o escopo atual (demo pública sem persistência em banco).
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# --- Barra lateral: escolha de modelo -----------------------------------
with st.sidebar:
    st.subheader("Configurações")
    model = st.selectbox(
        "Modelo (camada gratuita)",
        options=KNOWN_FREE_MODELS,
        index=0,
        help=(
            "'openrouter/free' escolhe automaticamente um modelo gratuito "
            "disponível — mais resiliente a mudanças no catálogo."
        ),
    )
    if st.button("Limpar conversa"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

# --- Histórico exibido ----------------------------------------------------
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Novo input do usuário ------------------------------------------------
user_input = st.chat_input("Digite sua mensagem...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Pensando..._")
        try:
            reply = chat_completion(st.session_state.messages, model=model)
            placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except LLMConfigError as exc:
            placeholder.error(f"Erro de configuração: {exc}")
        except LLMRequestError as exc:
            placeholder.warning(str(exc))

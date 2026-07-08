"""Cliente para a API da OpenRouter (compatível com o SDK da OpenAI).

Mantido separado da camada de UI (app.py) de propósito: a lógica de
comunicação com o modelo não deve depender do Streamlit, o que facilita
testar isoladamente e trocar de UI no futuro sem tocar aqui.
"""

import os

from openai import OpenAI, APIError, RateLimitError, AuthenticationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Roteador automático de modelos gratuitos: escolhe um modelo ":free"
# disponível no momento da chamada. É a opção mais resiliente, já que o
# catálogo de modelos gratuitos da OpenRouter muda com frequência.
DEFAULT_MODEL = "openrouter/free"

# Alguns modelos gratuitos populares para quem prefere escolher manualmente.
# A disponibilidade muda com o tempo — confira sempre a lista atual em
# https://openrouter.ai/models?fmt=cards&order=pricing-low-to-high
KNOWN_FREE_MODELS = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-coder:free",
]


class LLMConfigError(RuntimeError):
    """Erro de configuração (ex: chave de API ausente)."""


class LLMRequestError(RuntimeError):
    """Erro ao chamar a API da OpenRouter (rate limit, falha de rede, etc.)."""


def get_client() -> OpenAI:
    """Cria o cliente da OpenRouter usando a chave lida de variável de ambiente.

    A chave NUNCA deve ficar hardcoded no código-fonte. Localmente ela vem de
    um arquivo .env (fora do controle de versão); no Render, ela é definida
    como env var no painel do serviço.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "OPENROUTER_API_KEY não encontrada. Defina a variável de ambiente "
            "(veja .env.example) antes de rodar a aplicação."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def chat_completion(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Envia o histórico de mensagens para a OpenRouter e retorna o texto da resposta.

    `messages` segue o formato padrão OpenAI: [{"role": "user"/"assistant"/"system", "content": "..."}]
    """
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # Cabeçalhos opcionais recomendados pela OpenRouter para
            # identificar a aplicação nos rankings públicos de uso.
            extra_headers={
                "HTTP-Referer": os.environ.get("APP_URL", "https://localhost:8501"),
                "X-Title": "COSTAR Chat Agent",
            },
        )
    except AuthenticationError as exc:
        raise LLMConfigError(
            "Chave de API inválida ou expirada. Gere uma nova em openrouter.ai/keys."
        ) from exc
    except RateLimitError as exc:
        raise LLMRequestError(
            "Limite de requisições da camada gratuita atingido. Aguarde um "
            "minuto e tente novamente, ou troque para outro modelo :free."
        ) from exc
    except APIError as exc:
        raise LLMRequestError(f"Erro ao chamar a OpenRouter: {exc}") from exc

    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise LLMRequestError(
            "A API retornou uma resposta vazia. Tente novamente ou troque de modelo."
        )
    return content

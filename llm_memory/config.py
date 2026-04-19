import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def get_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not key:
        raise EnvironmentError(
            "[Config] OPENROUTER_API_KEY не задан. "
            "Создайте файл .env с OPENROUTER_API_KEY=<ваш ключ> или задайте переменную окружения."
        )
    return key


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_SMART = os.environ.get("MODEL_SMART", "z-ai/glm-4.5-air:free")
MODEL_FAST  = os.environ.get("MODEL_FAST",  "meta-llama/llama-3-8b-instruct:free")

SUMMARIZE_EVERY_N_MSG     = int(os.environ.get("SUMMARIZE_EVERY_N_MSG", "5"))
MAX_CONTEXT_TOKENS        = int(os.environ.get("MAX_CONTEXT_TOKENS", "4000"))

SUMMARIZE_TOKEN_THRESHOLD = int(os.environ.get("SUMMARIZE_TOKEN_THRESHOLD", "3000"))
KEEP_RECENT_K             = int(os.environ.get("KEEP_RECENT_K", "4"))
MAX_PROMPT_TOKENS         = int(os.environ.get("MAX_PROMPT_TOKENS", "8000"))
SUMMARIZE_MAX_MESSAGES    = int(os.environ.get("SUMMARIZE_MAX_MESSAGES", "50"))

MIN_SIMILARITY            = float(os.environ.get("MIN_SIMILARITY", "0.3"))
CHUNK_MAX                 = int(os.environ.get("CHUNK_MAX", "2000"))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

MEMORY_DIR = os.environ.get("MEMORY_DIR", "project_memory")

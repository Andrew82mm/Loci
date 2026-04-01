# colors.py — ANSI-цвета для терминала
# Использование: from colors import c, log
import sys

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# Цвета текста
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

# Яркие варианты
BRIGHT_RED     = "\033[91m"
BRIGHT_GREEN   = "\033[92m"
BRIGHT_YELLOW  = "\033[93m"
BRIGHT_BLUE    = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN    = "\033[96m"
BRIGHT_WHITE   = "\033[97m"

def _supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def c(text, *codes):
    """Оборачивает текст в ANSI-коды, если терминал поддерживает цвет."""
    if not _supports_color():
        return str(text)
    return "".join(codes) + str(text) + RESET

# ─── Готовые шаблоны логов ──────────────────────────────────────────────────

def log_system(msg):
    """[System] — голубой"""
    print(c(f"[System] {msg}", BRIGHT_CYAN))

def log_ok(msg):
    """Успех — зелёный"""
    print(c(f"  ✓ {msg}", BRIGHT_GREEN))

def log_warn(msg):
    """Предупреждение — жёлтый"""
    print(c(f"  ⚠ {msg}", BRIGHT_YELLOW))

def log_error(msg):
    """Ошибка — красный"""
    print(c(f"  ✗ {msg}", BRIGHT_RED), file=sys.stderr)

def log_knowledge(msg):
    """Граф знаний — маджента"""
    print(c(f"[Knowledge] {msg}", MAGENTA))

def log_rag(msg):
    """RAG — синий"""
    print(c(f"[RAG] {msg}", BLUE))

def log_llm(msg):
    """LLM-ошибка — ярко-красный"""
    print(c(f"[LLM Error] {msg}", BRIGHT_RED), file=sys.stderr)

def log_snapshot(msg):
    """Снэпшот — жёлтый"""
    print(c(f"[Snapshot] {msg}", YELLOW))

def separator(char="─", width=50):
    """Разделитель"""
    print(c(char * width, DIM))

def banner(title):
    """Шапка при запуске"""
    width = 52
    line  = "═" * width
    print(c(line, BRIGHT_CYAN))
    print(c(f"  {title}", BOLD + BRIGHT_WHITE))
    print(c(line, BRIGHT_CYAN))

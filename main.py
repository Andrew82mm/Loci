# main.py
import os
import sys
from core_engine import MemoryEngine
from colors import (
    banner, separator, log_system, log_ok, log_warn, log_error,
    c, BRIGHT_WHITE, BRIGHT_CYAN, BRIGHT_YELLOW, BRIGHT_GREEN,
    BRIGHT_MAGENTA, DIM, BOLD, GREEN, YELLOW, CYAN, MAGENTA
)

# ─── Справка ───────────────────────────────────────────────────────────────

HELP_TEXT = f"""
{c('Доступные команды:', BOLD + BRIGHT_WHITE)}

  {c('pin <текст>', BRIGHT_GREEN)}
      Закрепить важную информацию (высокий приоритет в контексте)

  {c('edit <имя файла>', BRIGHT_YELLOW)}
      Редактировать файл памяти вручную
      Примеры: edit pinned | edit context | edit <entity_name>

  {c('project <название>', BRIGHT_CYAN)}
      Переключить активный namespace проекта

  {c('snapshots', BRIGHT_MAGENTA)}
      Показать список снэпшотов памяти

  {c('rollback [имя]', BRIGHT_MAGENTA)}
      Откатить память к снэпшоту (без имени — к последнему)

  {c('reindex', BRIGHT_CYAN)}
      Принудительная переиндексация всей памяти

  {c('help', DIM)}
      Показать эту справку

  {c('exit / quit', DIM)}
      Выйти из программы
"""

# ─── Рендер ответа ─────────────────────────────────────────────────────────

def render_response(response: str, references: list[str]):
    """Выводит ответ ИИ с подсветкой секции References."""
    import re
    # Отделяем тело от References
    ref_pattern = re.compile(r"\nReferences:.*", re.IGNORECASE | re.DOTALL)
    body = ref_pattern.sub("", response).strip()

    print()
    print(c("AI:", BOLD + BRIGHT_WHITE), body)

    if references:
        refs_str = "  ".join(c(r, BRIGHT_CYAN) for r in references)
        print()
        print(c("  Источники: ", DIM) + refs_str)
    print()

# ─── Список снэпшотов ──────────────────────────────────────────────────────

def show_snapshots(engine: MemoryEngine):
    snaps = engine.list_snapshots()
    if not snaps:
        log_warn("Снэпшотов пока нет.")
        return

    separator("─", 52)
    print(c("  Снэпшоты памяти:", BOLD + BRIGHT_MAGENTA))
    separator("─", 52)
    for i, s in enumerate(snaps[:10]):  # показываем max 10
        label = s.get("label") or ""
        label_str = f"  [{label}]" if label else ""
        ts    = s.get("timestamp", "?")
        name  = s["name"]
        print(f"  {c(str(i+1).rjust(2), DIM)}. {c(ts, BRIGHT_YELLOW)}{c(label_str, DIM)}")
        print(f"      {c(name, DIM)}")
    separator("─", 52)

# ─── Редактор файла ────────────────────────────────────────────────────────

def inline_editor(fname: str, engine: MemoryEngine):
    print(c(f"\nРедактирование: {fname}", BRIGHT_YELLOW))
    print(c("Введите новое содержимое. Напишите", DIM),
          c("SAVE", BOLD + BRIGHT_GREEN),
          c("на отдельной строке для сохранения,", DIM),
          c("CANCEL", BOLD + BRIGHT_WHITE),
          c("для отмены.", DIM))
    lines = []
    while True:
        try:
            line = sys.stdin.readline()
        except EOFError:
            break
        stripped = line.strip()
        if stripped == "SAVE":
            engine.manual_edit(fname, "".join(lines))
            break
        elif stripped == "CANCEL":
            print(c("  Редактирование отменено.", DIM))
            break
        else:
            lines.append(line)

# ─── Главный цикл ──────────────────────────────────────────────────────────

def run_cli():
    banner("Local LLM Memory System  v0.2")
    print(c(f"  Память: {os.path.abspath('project_memory')}", DIM))
    print(c("  Напишите 'help' для справки по командам.", DIM))
    separator()

    try:
        engine = MemoryEngine()
        log_ok("Система инициализирована.")
    except EnvironmentError as e:
        log_error(str(e))
        sys.exit(1)
    except Exception as e:
        log_error(f"Ошибка инициализации: {e}")
        sys.exit(1)

    separator()

    while True:
        try:
            user_input = input(c("\nВы: ", BOLD + BRIGHT_GREEN)).strip()

            if not user_input:
                continue

            cmd = user_input.lower()

            # ── Выход ─────────────────────────────────────────────────────
            if cmd in ("exit", "quit", "q"):
                print(c("\nДо свидания!", BRIGHT_CYAN))
                break

            # ── Справка ───────────────────────────────────────────────────
            elif cmd == "help":
                print(HELP_TEXT)

            # ── Закрепление ───────────────────────────────────────────────
            elif cmd.startswith("pin "):
                engine.pin(user_input[4:].strip())

            # ── Редактирование ────────────────────────────────────────────
            elif cmd.startswith("edit "):
                fname = user_input[5:].strip()
                inline_editor(fname, engine)

            # ── Переключение проекта ──────────────────────────────────────
            elif cmd.startswith("project "):
                project = user_input[8:].strip()
                engine.storage.set_project(project)
                engine.rag.reindex_all()

            # ── Список снэпшотов ──────────────────────────────────────────
            elif cmd == "snapshots":
                show_snapshots(engine)

            # ── Откат ─────────────────────────────────────────────────────
            elif cmd.startswith("rollback"):
                parts = user_input.split(maxsplit=1)
                snap_name = parts[1].strip() if len(parts) > 1 else ""
                if not snap_name:
                    # Показываем список и просим выбрать
                    show_snapshots(engine)
                    choice = input(c("\n  Введите номер или имя снэпшота (Enter = последний): ",
                                     BRIGHT_MAGENTA)).strip()
                    if choice.isdigit():
                        snaps = engine.list_snapshots()
                        idx = int(choice) - 1
                        if 0 <= idx < len(snaps):
                            snap_name = snaps[idx]["name"]
                        else:
                            log_warn("Неверный номер.")
                            continue
                    else:
                        snap_name = choice  # пустая строка → последний

                ok = engine.rollback(snap_name or "")
                if ok:
                    log_ok("Откат выполнен успешно.")
                else:
                    log_warn("Откат не выполнен.")

            # ── Переиндексация ────────────────────────────────────────────
            elif cmd == "reindex":
                engine.rag.reindex_all()

            # ── Чат ───────────────────────────────────────────────────────
            else:
                print(c("  думаю...", DIM), end="\r", flush=True)
                response, references = engine.chat(user_input)
                render_response(response, references)

        except KeyboardInterrupt:
            print(c("\n  Прерывание. Напишите 'exit' для выхода.", DIM))
        except Exception as e:
            log_error(f"Неожиданная ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_cli()

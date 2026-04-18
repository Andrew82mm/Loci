# LLM Memory System — План улучшений

> Документ для исполнителя (модель/инженер). Задачи упорядочены по зависимостям: не перескакивать вперёд, пока не закрыт предыдущий блок.
> Каждая задача содержит: **что**, **зачем**, **где**, **как**, **acceptance criteria**.
> Правило: после каждого блока — прогнать тесты (`pytest`) и сделать коммит с префиксом `phase-N:`.

---

## Общие правила работы

1. **Не ломать обратную совместимость папки `project_memory/`** — у пользователя там могут быть реальные данные. Любая миграция — через отдельный скрипт `scripts/migrate_vN_to_vM.py`.
2. **Писать тесты параллельно коду.** Минимум: smoke-тест на каждую публичную функцию. Используй `pytest` + `tmp_path` fixture для изоляции.
3. **Тип-хинты обязательны** во всём новом коде. `mypy --strict` должен проходить для новых модулей.
4. **Никаких `print` в библиотечном коде** — только через `colors.py` логгеры. В будущем заменим на `logging`.
5. **Коммиты атомарные**, сообщение в формате `phase-N: <короткое описание>`.
6. **Не удалять рабочие модули, пока новые не заменят их полностью.** Параллельное существование → свитч одной строкой → удаление старого.

---

## Состояние проекта на момент написания плана

Существующие файлы (flat layout):
- `main.py` — CLI entrypoint, монолит с if-elif каскадом
- `core_engine.py` — `MemoryEngine`: god-object (buffer, chat, summarize, pin, rollback)
- `storage.py` — `StorageManager`: файлы, snapshot'ы, архив, mtime-index
- `rag_engine.py` — `RAGEngine`: Chroma + graph-expansion
- `knowledge_graph.py` — `KnowledgeGraph`: regex-based extraction в markdown
- `llm_client.py` — OpenRouter HTTP
- `config.py` — env vars, raise на импорте
- `colors.py` — цветной лог

Тестов нет. Структуры пакета нет. Цикл импорта (`rag_engine` импортит `knowledge_graph` внутри метода).

---

# PHASE 0 — Фундамент (обязательно перед всем остальным)

Цель: привести репозиторий к состоянию, в котором можно безопасно работать.

## 0.1. Инициализация инфраструктуры

**Задача:** создать базовую обвязку проекта.

**Действия:**
- [ ] Создать `pyproject.toml` (poetry или setuptools). Пакет назвать `llm_memory`.
- [ ] Добавить dev-зависимости: `pytest`, `pytest-cov`, `mypy`, `ruff`, `pydantic>=2`.
- [ ] Создать `.pre-commit-config.yaml` с `ruff` + `ruff-format` + `mypy` на новые файлы.
- [ ] Создать `tests/` с `conftest.py`, где определить fixture `tmp_memory_dir` (временная директория + инициализация `StorageManager`).
- [ ] Добавить `Makefile` с целями: `test`, `lint`, `typecheck`, `run`.
- [ ] `.gitignore`: добавить `.pytest_cache`, `.mypy_cache`, `.coverage`, `htmlcov/`, `project_memory/` (пользовательские данные не в репо).

**Acceptance:** `make test` запускается и проходит (пусть даже с 0 тестами).

## 0.2. Smoke-тесты на существующий код

**Задача:** зафиксировать текущее поведение до рефакторинга.

**Действия:** написать smoke-тесты в `tests/smoke/`:
- [ ] `test_storage_read_write.py`: `write_file` + `read_file` round-trip с frontmatter.
- [ ] `test_storage_snapshot.py`: create → modify → restore → проверить восстановление.
- [ ] `test_kg_extract.py`: мокнуть `llm_client`, передать заранее заготовленный ответ, проверить что создался правильный файл сущности.
- [ ] `test_rag_index.py`: создать 2 md-файла, вызвать `_sync_index()`, проверить что `collection.count() > 0`.
- [ ] `test_engine_chat.py`: мокнуть `llm_client`, проверить что buffer растёт и суммаризация триггерится.

**Как мокать LLM:** в `conftest.py`:
```python
@pytest.fixture
def mock_llm(monkeypatch):
    responses = []
    def fake_generate(model, sys, user, **kw):
        return responses.pop(0) if responses else "MOCK"
    monkeypatch.setattr("llm_client.llm_client.generate", fake_generate)
    return responses  # tests append to this list
```

**Acceptance:** все smoke-тесты зелёные, `pytest --cov` показывает ≥40% coverage существующего кода.

## 0.3. Убрать `raise` на импорте config

**Задача:** [config.py:14-18](config.py#L14-L18) падает с `EnvironmentError` при любом импорте без ключа. Это ломает тесты и dry-run.

**Действия:**
- [ ] Заменить `raise` на ленивую проверку: функция `get_openrouter_key()` делает raise только при первом вызове.
- [ ] В `llm_client.LLMClient.__init__` вызывать `get_openrouter_key()`.
- [ ] Тесты, не использующие LLM, больше не должны требовать ключ.

**Acceptance:** `OPENROUTER_API_KEY=""  python -c "import config"` не падает.

---

# PHASE 1 — MVP stabilization (критические баги + структура)

Цель: починить data-integrity баги, убрать god-object, довести до состояния "можно пользоваться без страха".

## 1.1. Структура пакета

**Задача:** перевести flat layout в пакет `llm_memory/`.

**Целевая структура:**
```
llm_memory/
  __init__.py
  config.py
  models.py              # pydantic: Fact, Entity, Message, Snapshot
  engine.py              # MemoryEngine (тонкая оркестрация)
  buffer.py              # ConversationBuffer (L1 buffer)
  summarizer.py          # SummarizationPipeline
  prompt_builder.py      # ContextAssembler + token budget
  storage/
    __init__.py
    base.py              # StorageBackend Protocol
    filesystem.py        # FilesystemBackend (текущее)
  graph/
    __init__.py
    extractor.py         # LLM → Fact
    resolver.py          # entity resolution
    index.py             # relations sqlite
    renderer.py          # Fact → markdown
  rag/
    __init__.py
    chunker.py
    vector.py            # Chroma wrapper
    retriever.py
  llm/
    __init__.py
    client.py            # OpenRouter
    tokens.py            # tiktoken wrapper
  cli/
    __init__.py
    main.py              # entry
    commands.py          # typer/click commands
  colors.py              # (оставить как есть, пока)
tests/
  smoke/
  unit/
  integration/
scripts/
  migrate_*.py
```

**Действия:**
- [ ] Переместить файлы, обновить импорты (`from llm_memory.storage.filesystem import ...`).
- [ ] Оставить в корне shim `main.py` с `from llm_memory.cli.main import run_cli; run_cli()` для обратной совместимости запуска.
- [ ] Прогнать smoke-тесты после каждого перемещения.

**Acceptance:** `python main.py` работает, все smoke-тесты проходят, структура соответствует целевой.

## 1.2. Models — pydantic-схемы

**Задача:** ввести типизированные модели данных. Всё остальное будет опираться на них.

**Файл:** `llm_memory/models.py`

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime

class Entity(BaseModel):
    name: str              # canonical
    aliases: list[str] = []
    file_path: str | None = None

class Fact(BaseModel):
    subject: str           # entity name
    predicate: str         # "implements", "uses", or "_attribute" for standalone
    object: str | None = None
    raw_text: str          # исходная строка для рендера в md
    source_chunk: str      # откуда извлечено
    extracted_at: datetime
    confidence: float = 1.0

class RetrievedChunk(BaseModel):
    content: str
    source: str            # relative path
    score: float
    reason: Literal["vector", "graph_hop", "pinned", "task", "summary"]

class Snapshot(BaseModel):
    name: str
    timestamp: datetime
    label: str = ""
    path: str
    includes_chroma: bool
    parent_snapshot: str | None = None   # для undo rollback
```

**Acceptance:** `pydantic` модели валидируются в тестах. Все остальные модули начинают использовать их вместо dict'ов.

## 1.3. КРИТИЧНО: починить потерю данных при фейле суммаризации

**Проблема:** [core_engine.py:92-161](core_engine.py#L92-L161) — если summarize failed, buffer всё равно чистится на строке 159.

**Действия в `summarizer.py`:**
- [ ] Переставить порядок: **сначала архивация**, потом summarize, потом (только при успехе) очистка.
- [ ] Ввести концепцию "успеха цикла": summarize считается успешным, только если получили валидный summary (не `Error:` и не пустой).
- [ ] При фейле: пометить архив-файл `failed_summarization: true` в frontmatter, **не чистить buffer**, залогировать warning, вернуть управление.
- [ ] При следующем вызове `chat()` повторная попытка суммаризации произойдёт автоматически (buffer всё ещё превышает лимит).

**Псевдокод:**
```python
def run_cycle(self, buffer: list[Message]) -> CycleResult:
    archive_path = self.storage.archive(buffer, pending=True)
    snapshot = self.storage.create_snapshot(label="pre_summarize")
    try:
        task = self._update_task(buffer)
        summary = self._summarize(buffer)
        if not summary or summary.startswith("Error:"):
            raise SummarizationError("empty or errored summary")
        facts = self.extractor.extract(summary)
        self._persist(task, summary, facts)
        self.storage.archive_finalize(archive_path)  # снимает pending
        return CycleResult(ok=True, clear_buffer=True)
    except Exception as e:
        log_warn(f"Summarization failed: {e}. Buffer preserved.")
        self.storage.restore_snapshot(snapshot.name, silent=True)
        return CycleResult(ok=False, clear_buffer=False)
```

**Acceptance:**
- Unit-тест: мокнутый LLM возвращает `Error:`, buffer не чистится.
- Unit-тест: мокнутый LLM возвращает пустую строку, buffer не чистится.
- Unit-тест: нормальный успешный путь — buffer чистится, архив-файл без `pending`.

## 1.4. КРИТИЧНО: Sliding window buffer

**Проблема:** после суммаризации buffer обнуляется полностью → модель теряет свежий контекст.

**Действия в `buffer.py`:**
- [ ] `ConversationBuffer` класс: `add(msg)`, `all() -> list[Message]`, `recent(k) -> list[Message]`, `to_summarize() -> list[Message]`, `clear_summarized(keep_last_k: int)`.
- [ ] `to_summarize()` возвращает все сообщения, кроме последних K (по умолчанию `KEEP_RECENT_K=4` из config).
- [ ] После успешного цикла буфер сокращается до последних K сообщений, а не очищается полностью.

**Acceptance:** unit-тест: добавить 20 сообщений, триггернуть summarize, проверить что buffer содержит последние 4.

## 1.5. КРИТИЧНО: Token-based trigger суммаризации

**Проблема:** [core_engine.py:93](core_engine.py#L93) — триггер по количеству сообщений. Одно сообщение на 20K токенов = OOM промпта.

**Действия:**
- [ ] `llm/tokens.py`: функции `count_tokens(text: str, model: str) -> int`, используя `tiktoken` с fallback на character-count/4 для моделей без tokenizer.
- [ ] В `config.py`: `SUMMARIZE_TOKEN_THRESHOLD = 3000`, `KEEP_RECENT_K = 4`, `MAX_PROMPT_TOKENS = 8000`.
- [ ] В `engine.chat()`: после добавления сообщения вызывать `buffer.total_tokens()` → если > threshold, запустить `summarizer.run_cycle()`.
- [ ] Сохранить message-based trigger как secondary: max-messages = 50 (safety).

**Acceptance:** unit-тест: добавить одно сообщение с 5000 токенов → summarize триггерится. Добавить 10 коротких → не триггерится.

## 1.6. КРИТИЧНО: Chroma в snapshot + atomic rollback

**Проблема:** [storage.py:107-133](storage.py#L107-L133) не копирует Chroma DB. Rollback восстанавливает markdown, но vector-index остаётся в будущем → неконсистентность. `reindex_all()` в [core_engine.py:212](core_engine.py#L212) это маскирует, но дорого.

**Действия в `storage/filesystem.py`:**
- [ ] В `create_snapshot()`: дополнительно `shutil.copytree(paths["chroma_db"], snap_path/"chroma_db")`. Перед копированием — `chroma_client.reset()` или вызов persist (зависит от версии Chroma; проверить документацию).
- [ ] Копировать также `_system/conversation_buffer.json` и `_system/relations.*` (когда появится Phase 1.9).
- [ ] В `restore_snapshot()`: восстанавливать Chroma директорию + пересоздавать клиент (`client = chromadb.PersistentClient(...)` заново). В `RAGEngine` добавить метод `reload_client()`.
- [ ] Убрать вызов `reindex_all()` из `MemoryEngine.rollback` — это костыль.
- [ ] `before_restore` snapshot должен содержать `parent_snapshot` в meta.json → поддержка undo rollback.

**Acceptance:**
- Integration-тест: write → snapshot A → write → snapshot B → restore A → проверить что Chroma содержит только данные снэпшота A.
- Тест undo: restore A → undo → состояние возвращается к B.

## 1.7. Score threshold в RAG + distances

**Проблема:** [rag_engine.py:106](rag_engine.py#L106) игнорирует distances → возвращаются нерелевантные результаты.

**Действия в `rag/retriever.py`:**
- [ ] `query(..., include=["distances", "metadatas", "documents"])`.
- [ ] Фильтр: Chroma возвращает squared-L2 distance для `all-MiniLM` по умолчанию → конвертировать в cosine-score; порог `MIN_SIMILARITY = 0.3` в config.
- [ ] Возвращать `RetrievedChunk` с `score` и `reason="vector"`.
- [ ] Логировать сколько отсеяно (`log_rag("filtered X/Y chunks below threshold")`).

**Acceptance:** unit-тест: запрос без релевантных данных возвращает пустой список, не 5 мусорных результатов.

## 1.8. Markdown-aware chunking

**Проблема:** [rag_engine.py:84-94](rag_engine.py#L84-L94) режет по 800 символов fixed-stride.

**Действия в `rag/chunker.py`:**
- [ ] Использовать `marko` или `mistune` для парсинга.
- [ ] Стратегия: сплит по заголовкам (`#`, `##`, `###`). Если секция > `CHUNK_MAX` (2000 символов) → вторичный сплит по параграфам. Если и это не хватает → fallback на sliding window 800/100.
- [ ] Каждый чанк получает мета: `heading_path: list[str]` (breadcrumbs), `file: str`.
- [ ] Для файлов без заголовков (как `pinned.md`) — сплит по `\n\n` (параграфы).

**Acceptance:**
- Unit-тест: файл с 3 секциями → 3 чанка, каждый содержит заголовок секции.
- Unit-тест: длинная секция разбивается правильно, без потери данных.

## 1.9. Explicit relations index

**Проблема:** граф сейчас — regex по markdown. `get_connected_nodes` — O(файл-чтение × каждый запрос).

**Действия в `graph/index.py`:**
- [ ] SQLite-таблица:
  ```sql
  CREATE TABLE relations (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT,
    source_file TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    extracted_at TEXT NOT NULL,
    UNIQUE(subject, predicate, object)
  );
  CREATE INDEX idx_subject ON relations(subject);
  CREATE INDEX idx_object ON relations(object);
  ```
- [ ] API: `add(fact: Fact)`, `neighbors(entity: str, direction: Literal["out","in","both"] = "both") -> list[str]`, `query(subject=None, predicate=None, object=None) -> list[Fact]`.
- [ ] Двунаправленность: `neighbors("A", "both")` возвращает и исходящие и входящие связи.
- [ ] Markdown остаётся как человекочитаемое представление (renderer генерирует его из индекса).

**Acceptance:**
- Unit-тест: добавить факт `A uses B` → `neighbors("A") == ["B"]`, `neighbors("B") == ["A"]`.
- Unit-тест: при rollback индекс восстанавливается из snapshot.

## 1.10. Мультиязычный эмбеддинг

**Проблема:** [rag_engine.py:17-19](rag_engine.py#L17-L19) использует английскую модель.

**Действия:**
- [ ] В config: `EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"`.
- [ ] При смене модели **нужна переиндексация** — добавить в `RAGEngine.__init__` проверку: сохранять имя модели в meta-файле Chroma-директории. Если не совпадает с текущим → `reindex_all()` автоматически.

**Acceptance:** русскоязычный тестовый корпус → релевантный результат в top-3 (ручная проверка или закладочный тест с golden set).

## 1.11. Прибить циклический импорт

**Проблема:** [rag_engine.py:121-122](rag_engine.py#L121-L122) импортирует KG внутри метода.

**Действия:**
- [ ] В `rag/retriever.py` принимать `GraphIndex` через конструктор: `Retriever(vector_store, graph_index)`.
- [ ] DI в `engine.py`: создаётся единый `GraphIndex`, передаётся и в `Retriever`, и в `Extractor`.

**Acceptance:** `ruff` / `mypy` не ругается, импорты только на верхнем уровне.

---

# PHASE 2 — Production-ready

Цель: надёжность, масштабируемость, мультисессионность, контроль качества извлечения.

## 2.1. Transactional writer (WAL)

**Проблема:** падение процесса между `write_file` и `index_file` → рассинхронизация markdown и Chroma.

**Действия в `storage/filesystem.py`:**
- [ ] `_system/wal.jsonl` — append-only лог намерений: `{"op": "write", "path": "...", "content_hash": "...", "status": "pending"}` → `"status": "committed"`.
- [ ] При старте `StorageManager`: прочитать WAL, найти `pending` записи, реапплаить или откатить.
- [ ] `write_file` обёртка: write WAL pending → write file → update index → update Chroma → mark WAL committed.
- [ ] Compaction WAL: при старте, если > N entries — сжать в backup-файл.

**Acceptance:** integration-тест: симулировать падение после записи файла, но до chroma update → recovery восстанавливает целостность.

## 2.2. Git-backed snapshots

**Проблема:** full-copy снэпшоты → диск.

**Действия:**
- [ ] `storage/git_backed.py`: `GitBackedStorage(FilesystemStorage)`.
- [ ] При init: `git init` в `project_memory/`, `.gitignore` для `_system/chroma_db/` и `_system/wal.jsonl`.
- [ ] `create_snapshot()` = `git add -A && git commit -m "<label>"`, возвращает SHA.
- [ ] `restore_snapshot(sha)` = `git checkout <sha> -- .` + отдельно восстановление Chroma из tarball в `_system/chroma_snapshots/<sha>.tar.gz`.
- [ ] `list_snapshots()` = `git log --format=...`.
- [ ] GC: функция `prune_snapshots(keep_last=50, keep_tagged=True)`.

**Acceptance:** все тесты snapshot/rollback проходят с Git-backend. Размер репо не растёт линейно от количества снэпшотов.

## 2.3. Entity resolution

**Проблема:** `[[Андрей]]` vs `[[Andrey]]` — два файла.

**Действия в `graph/resolver.py`:**
- [ ] Таблица aliases в SQLite:
  ```sql
  CREATE TABLE entities (name TEXT PRIMARY KEY, canonical TEXT, embedding BLOB);
  CREATE INDEX idx_canonical ON entities(canonical);
  ```
- [ ] `resolve(name: str) -> Entity`:
  1. Exact match после нормализации (lower, trim, NFC).
  2. Alias lookup.
  3. Top-K nearest by embedding (threshold > 0.85).
  4. Иначе создать новый canonical.
- [ ] В `extractor.py`: перед вызовом LLM для извлечения — передать в промпт список известных canonical names: "Known entities: ...". LLM должна использовать их.

**Acceptance:** unit-тесты на все 4 случая разрешения. Integration: 2 суммаризации с вариантами имени → один файл.

## 2.4. Structured LLM extraction

**Проблема:** [knowledge_graph.py:29-41](knowledge_graph.py#L29-L41) — сырой текст → regex. LLM может галлюцинировать.

**Действия в `graph/extractor.py`:**
- [ ] Использовать function calling / JSON mode OpenRouter (проверить поддержку для конкретных моделей).
- [ ] Schema:
  ```python
  class FactsExtraction(BaseModel):
      facts: list[Fact]
  ```
- [ ] Промпт: "Extract facts as JSON matching schema. Return empty list if uncertain."
- [ ] Fallback: если JSON mode не поддерживается → промпт "return only JSON" + `json.loads` с retry через `json-repair`.
- [ ] Валидация pydantic, невалидные факты логируются и отбрасываются.

**Acceptance:** unit-тест с мокнутым LLM, возвращающим невалидный JSON → extractor возвращает пустой список, не падает.

## 2.5. LLM-as-judge validation

**Действия:**
- [ ] После extract — для каждого факта вызов "judge" LLM (MODEL_SMART, batch): "Is this fact supported by source chunk? yes/no".
- [ ] Отбрасывать факты с `no`.
- [ ] Опционально: включается через `config.ENABLE_FACT_VALIDATION = True`, по умолчанию off (дорого).

**Acceptance:** unit-тест с мокнутым judge, возвращающим `no` на все → facts не сохраняются.

## 2.6. Conflict detection

**Действия в `graph/index.py`:**
- [ ] При `add(fact)`: искать существующие факты с тем же `(subject, predicate)`. Если объект отличается → помечать оба `contested=true`.
- [ ] Опционально (Phase 2.5): LLM-рассуждение "которая версия более актуальна" на основе timestamp и source.

**Acceptance:** unit-тест: добавить "A works_at B" и "A works_at C" → оба помечены contested.

## 2.7. Per-namespace Chroma collections

**Проблема:** [rag_engine.py:20-23](rag_engine.py#L20-L23) — одна коллекция. Смена project = полная reindex.

**Действия:**
- [ ] `RAGEngine` держит dict `{namespace: Collection}`.
- [ ] `set_namespace(name)` — get-or-create коллекцию `memory_{name}`.
- [ ] Global knowledge (`_global`) — отдельная всегда-активная коллекция, поиск merge'ит её результаты с текущим проектом.

**Acceptance:** switch между проектами O(1), не триггерит reindex.

## 2.8. Token budget в prompt assembly

**Действия в `prompt_builder.py`:**
- [ ] `ContextAssembler.build(query: str, budget: int) -> str`.
- [ ] Приоритеты: `pinned` (всегда включать) → `task` → `summary` → `rag_hits` (top по score, усечение до budget).
- [ ] Каждый блок обрезается до N токенов с предупреждением "[truncated]".
- [ ] Budget по умолчанию = `MAX_PROMPT_TOKENS - estimated_response_tokens - user_input_tokens`.

**Acceptance:** unit-тест: передать 50 больших rag_hits → итоговый prompt ≤ budget.

## 2.9. BM25 + rerank

**Действия:**
- [ ] `rag/bm25.py`: `rank_bm25.BM25Okapi` на тех же чанках. Индекс пересоздаётся при изменении файлов.
- [ ] `retriever.retrieve()`: объединяет top-K от vector + top-K от BM25 (RRF — reciprocal rank fusion), дедуп по source+chunk.
- [ ] Опционально: cross-encoder rerank через `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) для финального top-N.

**Acceptance:** на заранее подготовленном golden set recall@5 увеличивается vs только vector.

## 2.10. File locking

**Действия:**
- [ ] `pip install filelock`.
- [ ] Оборачивать все write-операции в `FileLock(paths["system"] / ".lock")`.
- [ ] При старте если lock уже держится — лог warning "another session detected".

**Acceptance:** запуск двух процессов → второй ждёт или безопасно работает в read-only.

## 2.11. Snapshot retention

**Действия:**
- [ ] Config: `SNAPSHOT_KEEP_LAST = 50`, `SNAPSHOT_KEEP_AGE_DAYS = 30`.
- [ ] При `create_snapshot`: вызывать `_prune()`. Не удалять снэпшоты с `label` в `{"manual", "keep"}`.
- [ ] CLI команда `snapshot tag <name> <label>`.

**Acceptance:** после 60 снэпшотов — в директории 50 (+tagged).

## 2.12. CLI refactor на typer

**Проблема:** [main.py:125-203](main.py#L125-L203) — if-elif каскад, не тестируем.

**Действия в `cli/commands.py`:**
- [ ] `typer` app с командами: `chat` (default REPL), `pin`, `edit`, `project`, `snapshots`, `rollback`, `reindex`, `tag-snapshot`, `stats`.
- [ ] Каждая команда — функция, тестируемая через `typer.testing.CliRunner`.
- [ ] REPL-режим остаётся (команда без аргумента открывает чат).

**Acceptance:** unit-тест каждой CLI команды через CliRunner.

---

# PHASE 3 — Advanced features

Цель: конкурентные преимущества, требующие существенной работы.

## 3.1. Episodic vs semantic memory

**Концепция:** разделить "что произошло" (эпизод) и "что верно сейчас" (знание).

**Действия:**
- [ ] `archive/` → episodic memory, с embedding'ами по эпизодам. Search по эпизодам отдельно от semantic KG.
- [ ] Периодическая consolidation (cron-like джоб): старые эпизоды → агрегированные knowledge-facts, эпизоды можно удалять/сжимать.

## 3.2. Temporal knowledge graph

**Действия:**
- [ ] В таблице `relations` добавить `valid_from`, `valid_to`.
- [ ] Запросы: "где работал X в 2023" — временной фильтр.
- [ ] При conflict detection: новый факт с `valid_from=now` закрывает старый (`valid_to=now`), не помечая contested.

## 3.3. Multi-hop retrieval (GraphRAG-style)

**Действия:**
- [ ] `retriever.retrieve_multi_hop(query, max_hops=2)`: LLM-планировщик разбивает query на подзапросы, каждый запускает vector search, результаты объединяются.
- [ ] Альтернатива: HippoRAG-подход — PageRank по графу со стартовыми точками из vector-hits.

## 3.4. Forgetting curve

**Действия:**
- [ ] Каждому факту: `score = f(recency, access_count, pinned, confidence)`.
- [ ] Bg-job: факты со score < threshold → архивируются (убираются из активного индекса, остаются в snapshots).
- [ ] При поиске с флагом `--include-archived` — ищет и в архиве.

## 3.5. Obsidian plugin

**Действия:**
- [ ] Отдельный репо `obsidian-llm-memory`.
- [ ] Плагин: side-panel с RAG-hits для текущего выделения, кнопка "ask" открывает чат с контекстом.
- [ ] Backend: локальный HTTP сервер (FastAPI) в main проекте.

## 3.6. Web UI

**Действия:**
- [ ] FastAPI backend с REST + WebSocket.
- [ ] Frontend: SvelteKit/React, визуализация графа через `cytoscape.js`.
- [ ] Deploy: Docker-compose.

## 3.7. Local LLM (Ollama)

**Действия:**
- [ ] `llm/ollama_client.py` с тем же интерфейсом что `OpenRouterClient`.
- [ ] Config: `LLM_PROVIDER = "ollama" | "openrouter"`.
- [ ] Автодетект запущенного ollama через `GET /api/tags`.

## 3.8. Evaluation harness

**Действия:**
- [ ] `eval/` директория с бенчмарками: LongMemEval / LoCoMo subsets.
- [ ] Метрики: recall@K, precision@K, summarization ROUGE, fact F1.
- [ ] CI job: регрессия > 5% блокирует merge.

---

# Backlog (идеи без приоритета)

- Async I/O всех LLM вызовов.
- Event bus для extract → index → persist (Redis Streams / local async queue).
- CRDT layer для team collaboration.
- Encrypted at rest (age/gpg) для sensitive namespaces.
- MCP server integration (чтобы Claude Code/etc. могли использовать эту память напрямую).

---

# Порядок работы для исполнителя

1. **Начинай строго с Phase 0.** Без инфраструктуры и тестов ничего не делать.
2. **В рамках фазы** — по порядку задач (1.1 → 1.2 → ...). Зависимости указаны явно в "Действиях".
3. **После каждой задачи:**
   - прогнать `make test`
   - прогнать `make lint` и `make typecheck` для новых файлов
   - коммит `phase-N.M: <описание>`
4. **Если задача блокирована** (внешняя зависимость, неясные требования) — отметить в этом файле `BLOCKED: <причина>` и переходить к следующей независимой.
5. **Не добавлять новые задачи в план без апрува.** Если нашёл баг вне плана — создать issue/TODO и продолжать.
6. **Никогда не коммитить `project_memory/`** — там пользовательские данные.
7. **Миграции данных** — отдельные скрипты в `scripts/`, с dry-run режимом и бэкапом.

# Чек-лист перед закрытием каждой фазы

- [ ] Все задачи фазы закрыты или явно перенесены
- [ ] Coverage ≥ 70% для Phase 1, ≥ 80% для Phase 2
- [ ] `mypy --strict llm_memory/` без ошибок
- [ ] `ruff check llm_memory/` чистый
- [ ] CHANGELOG.md обновлён
- [ ] Ручной прогон сценария "чат → суммаризация → rollback → reindex" успешен

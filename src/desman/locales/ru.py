"""Russian. Plurals have three forms: 1 ключ, 2 ключа, 5 ключей (and 21 ключ, 11 ключей)."""

from __future__ import annotations

from desman.locales import Catalog, Forms


def plural(n: int) -> int:
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return 0
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return 1
    return 2


MESSAGES: dict[str, Forms] = {
    # ── Command help ─────────────────────────────────────────────────────────
    "DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)": (
        "DESMAN(P) — децентрализованная самоэволюционирующая модульная сеть агентов (прототип)"
    ),
    "Show help for commands.": "Показать справку по командам.",
    "Show the version and exit.": "Показать версию и выйти.",
    "{prog}, version {version}": "{prog}, версия {version}",
    "Create the database or apply pending migrations. Safe to re-run.": (
        "Создать базу данных или применить новые миграции. Можно запускать повторно."
    ),
    "Show the full history of TASK (its id or a unique prefix of it).\n\n"
    "Every status change, reviewer verdict, checkpoint and discarded piece of work, in order.\n"
    "Exits with 1 if the history does not account for the current state.": (
        "Показать всю историю задачи TASK (её id или уникальное начало id).\n\n"
        "По порядку: каждая смена статуса, вердикт ревьюера, checkpoint и выброшенная работа. "
        "Код выхода 1, если история не объясняет текущее состояние."
    ),
    "Continue TASK where it stopped: after a crash, a pause or the operator's decisions.\n\n"
    "Work a crashed run left behind is taken back at once if that run was on this machine,\n"
    "otherwise when its lease expires. Finished subtasks are never redone. Same exit codes as\n"
    "`run`.": (
        "Продолжить задачу TASK с места остановки: после падения, паузы или решений оператора.\n\n"
        "Работу упавшего запуска на этой машине забираем сразу, иначе — когда истечёт его "
        "аренда. Готовые подзадачи не переделываются. Коды выхода те же, что у `run`."
    ),
    "Show TASK and its subtasks. Same exit codes as `run`.": (
        "Показать задачу TASK и её подзадачи. Коды выхода те же, что у `run`."
    ),
    "List tasks, newest first.": "Список задач, новые сверху.",
    "List subtasks waiting for the operator (needs_human, failed), with the reason.": (
        "Подзадачи, которые ждут оператора (needs_human, failed), с причиной."
    ),
    "Send SUBTASK back to work with a fresh attempt budget.\n\n"
    "Works on a subtask waiting for you or one marked failed (its dependents are unblocked).\n"
    "Continue the task with `desman resume`.\n\n"
    'SUBTASK is "<task>/<number>" as `desman status` and `inbox` show it, or its id.': (
        "Вернуть подзадачу SUBTASK в работу с новым запасом попыток.\n\n"
        "Подходит для подзадачи, которая ждёт вас или помечена failed (зависящие от неё "
        "разблокируются). Продолжить задачу: `desman resume`.\n\n"
        'SUBTASK — это "<задача>/<номер>", как в `desman status` и `inbox`, или её id.'
    ),
    "Mark SUBTASK done as it is (its latest result is kept only if it was under review).\n\n"
    'SUBTASK is "<task>/<number>" as `desman status` and `inbox` show it, or its id.': (
        "Отметить подзадачу SUBTASK готовой как есть (последний результат сохраняется, только "
        "если он был на ревью).\n\n"
        'SUBTASK — это "<задача>/<номер>", как в `desman status` и `inbox`, или её id.'
    ),
    "Give up on SUBTASK: it becomes failed and everything depending on it is blocked.\n\n"
    "`desman retry` brings it back later.\n\n"
    'SUBTASK is "<task>/<number>" as `desman status` and `inbox` show it, or its id.': (
        "Отказаться от подзадачи SUBTASK: она становится failed, всё зависящее от неё "
        "блокируется.\n\n"
        "`desman retry` вернёт её позже.\n\n"
        'SUBTASK — это "<задача>/<номер>", как в `desman status` и `inbox`, или её id.'
    ),
    "Cancel TASK: every subtask not finished yet stops. A running one notices at its next\n"
    "heartbeat.": (
        "Отменить задачу TASK: все незавершённые подзадачи останавливаются. Выполняющаяся "
        "заметит это на ближайшем heartbeat."
    ),
    "Database file (default: $DESMAN_DATA_DIR/desman.db or ~/.local/share/desman/desman.db).": (
        "Файл базы данных (по умолчанию: $DESMAN_DATA_DIR/desman.db или "
        "~/.local/share/desman/desman.db)."
    ),
    "Database file with the call log (default: the usual one).": (
        "Файл базы данных с логом вызовов (по умолчанию — обычный)."
    ),
    "Machine-readable output.": "Вывод для программ (JSON).",
    "Why it failed (recorded in the audit log).": "Почему не получилось (пишется в аудит).",
    "Act only if the subtask has not changed since `inbox --json` showed this version.": (
        "Действовать, только если подзадача не менялась с версии, которую показал `inbox --json`."
    ),
    "Why it is acceptable.": "Почему это приемлемо.",
    "Guidance for the next attempt (passed to the executor).": (
        "Подсказка для следующей попытки (передаётся исполнителю)."
    ),
    "Only tasks with this status.": "Только задачи с этим статусом.",
    "How long a claimed subtask stays ours without a heartbeat; after a crash, `resume` waits "
    "at most this long.": (
        "Сколько захваченная подзадача остаётся нашей без heartbeat; после падения `resume` "
        "ждёт не дольше этого."
    ),
    "Sleep through provider pauses instead of returning (exit code 4) when only paused work is "
    "left.": (
        "Пережидать паузы провайдера, а не выходить (код 4), когда осталась только работа на паузе."
    ),
    "Drive the agents with a scripted fake provider (JSON script, see desman.llm.fake). "
    "Scripted latency and timeouts take real time.": (
        "Запустить агентов на фальшивом провайдере по сценарию (JSON, см. desman.llm.fake). "
        "Задержки и таймауты из сценария идут в реальном времени."
    ),
    "Provider to use, e.g. groq (default: the first configured one in the preference order). "
    "All its keys are used in turn.": (
        "Провайдер, например groq (по умолчанию: первый настроенный в порядке предпочтения). "
        "Все его ключи используются по очереди."
    ),
    "Model to use (default: the provider's free-tier default, see `desman keys check`).": (
        "Модель (по умолчанию: бесплатная модель провайдера, см. `desman keys check`)."
    ),
    "A task graph as JSON (TaskSpec: goal, priority, subtasks with depends_on).": (
        "Граф задачи в JSON (TaskSpec: goal, priority, subtasks с depends_on)."
    ),
    "Overrides the priority (default: normal, or the spec's).": (
        "Задаёт приоритет (по умолчанию: normal или из spec)."
    ),
    "Model calls and tokens spent, per provider and key (from the call log).": (
        "Вызовы моделей и потраченные токены по провайдерам и ключам (из лога вызовов)."
    ),
    "Window: e.g. 30m, 24h, 7d, or 'all'.": "Период: например 30m, 24h, 7d или 'all'.",
    "Configure DESMAN: providers, API keys, permissions (`show`, `reset`).": (
        "Настроить DESMAN: провайдеры, API-ключи, разрешения (`show`, `reset`)."
    ),
    "Show the configuration with API keys masked.": "Показать настройки со скрытыми API-ключами.",
    "Delete the configuration file, API keys included.": "Удалить файл настроек вместе с API-ключами.",
    "Do not ask for confirmation.": "Не спрашивать подтверждения.",
    "API keys: list, add, remove, import (always shown masked).": (
        "API-ключи: list, add, remove, import (всегда показываются скрытыми)."
    ),
    "How many keys each provider has, and what each spent in the last 24 hours.": (
        "Сколько ключей у каждого провайдера и сколько каждый потратил за 24 часа."
    ),
    "Ask each provider whether its keys work and what limits it reports. Spends no tokens.": (
        "Спросить у каждого провайдера, работают ли ключи и какие у них лимиты. Токены не тратятся."
    ),
    "Add an API key to PROVIDER (id, name or variable, e.g. groq). Several per provider.": (
        "Добавить API-ключ провайдеру PROVIDER (id, имя или переменная, например groq). "
        "Можно несколько."
    ),
    "A name to tell this key apart, e.g. 'work'.": "Название, чтобы отличать ключ, например 'work'.",
    "Read the key from standard input.": "Прочитать ключ из стандартного ввода.",
    "Remove key NUMBER (as `desman keys` lists it) of PROVIDER, or --all of them.": (
        "Удалить ключ номер NUMBER (как в `desman keys`) у провайдера PROVIDER, или все (--all)."
    ),
    "Remove every key of PROVIDER.": "Удалить все ключи провайдера PROVIDER.",
    "Add the keys in FILE (JSON or NAME=value lines) to the config. Keys already saved and\n"
    "names no provider uses are skipped; values are never printed.": (
        "Добавить в настройки ключи из файла FILE (JSON или строки NAME=value). Уже сохранённые "
        "ключи и имена, которые не использует ни один провайдер, пропускаются; значения не "
        "выводятся."
    ),
    # ── Tasks ────────────────────────────────────────────────────────────────
    "no database at {path} (nothing has run yet?)": (
        "нет базы данных {path} (ещё ничего не запускали?)"
    ),
    "{path}: schema v{version} ({n} migration applied)": (
        "{path}: схема v{version} (применена {n} миграция)",
        "{path}: схема v{version} (применено {n} миграции)",
        "{path}: схема v{version} (применено {n} миграций)",
    ),
    "Task {id}  {status}": "Задача {id}  {status}",
    "Task {id}  {status}  ({done}/{total} done)": "Задача {id}  {status}  (готово {done}/{total})",
    "Goal: {goal}": "Цель: {goal}",
    "History is incomplete:": "История неполна:",
    "give either GOAL or --spec, not both": "укажите либо GOAL, либо --spec, но не оба",
    "Task {id} created ({n} subtask)": (
        "Задача {id} создана ({n} подзадача)",
        "Задача {id} создана ({n} подзадачи)",
        "Задача {id} создана ({n} подзадач)",
    ),
    "GOAL is empty": "GOAL пустая",
    "{path}: not a valid task spec:\n{error}": "{path}: это не описание задачи:\n{error}",
    "Cannot use the tool policy: {error}": "Не удаётся использовать политику инструментов: {error}",
    "--fake-llm replaces the provider: drop --provider/--model": (
        "--fake-llm заменяет провайдера: уберите --provider/--model"
    ),
    "{path}: not a valid fake LLM script:\n{error}": (
        "{path}: это не сценарий фальшивой LLM:\n{error}"
    ),
    "Using {provider} · {model} ({n} key)": (
        "Используется {provider} · {model} ({n} ключ)",
        "Используется {provider} · {model} ({n} ключа)",
        "Используется {provider} · {model} ({n} ключей)",
    ),
    "fake LLM script ran out: {error}": "сценарий фальшивой LLM закончился: {error}",
    "Paused by a provider limit until {until}. Continue with: desman resume {ref}": (
        "Пауза из-за лимита провайдера до {until}. Продолжить: desman resume {ref}"
    ),
    "after {deps}": "после {deps}",
    "attempts q{quality}/t{transient}": "попытки q{quality}/t{transient}",
    "held by {owner} until {until}": "у {owner} до {until}",
    "paused until {until}": "пауза до {until}",
    "Nothing is waiting for you.": "Ничего не ждёт вашего решения.",
    "Decide with: desman retry|accept|fail <ref> [--note TEXT]": (
        "Решение: desman retry|accept|fail <ref> [--note ТЕКСТ]"
    ),
    "say why it failed": "напишите, почему не получилось",
    "it changed meanwhile, try again: {error}": "за это время что-то изменилось, повторите: {error}",
    "Task {id}: {n} subtask cancelled, task is {status}": (
        "Задача {id}: отменена {n} подзадача, статус задачи — {status}",
        "Задача {id}: отменено {n} подзадачи, статус задачи — {status}",
        "Задача {id}: отменено {n} подзадач, статус задачи — {status}",
    ),
    "[{seq}] changed since version {seen} (now {version}, {status}): look again with "
    "`desman inbox`": (
        "[{seq}] изменилась после версии {seen} (сейчас {version}, {status}): посмотрите снова "
        "в `desman inbox`"
    ),
    "[{seq}] is {status}: the operator can move it to {options}, not {target}": (
        "[{seq}] в статусе {status}: оператор может перевести её в {options}, но не в {target}"
    ),
    "nothing": "никуда",
    "it changed meanwhile, look again: {error}": (
        "за это время что-то изменилось, посмотрите снова: {error}"
    ),
    "Task is active again. Continue with: desman resume {ref}": (
        "Задача снова активна. Продолжить: desman resume {ref}"
    ),
    "Task is {status}.": "Статус задачи — {status}.",
    # ── Providers ────────────────────────────────────────────────────────────
    "no provider is configured: add a key with `desman keys add PROVIDER`": (
        "ни один провайдер не настроен: добавьте ключ через `desman keys add PROVIDER`"
    ),
    "{name} is local: local models are not supported yet": (
        "{name} — локальный провайдер: локальные модели пока не поддерживаются"
    ),
    "{name} is not configured: add a key with `desman keys add {id}`": (
        "{name} не настроен: добавьте ключ через `desman keys add {id}`"
    ),
    "{name} is not configured: add a key with `desman keys add {id}` and set its settings in "
    "`desman config`": (
        "{name} не настроен: добавьте ключ через `desman keys add {id}` и заполните его "
        "настройки в `desman config`"
    ),
    "only paid providers are configured ({names}), and DESMAN never picks a paid one by "
    "itself: choose it with --provider, e.g. --provider {id}": (
        "настроены только платные провайдеры ({names}), а платного DESMAN сам не выбирает: "
        "укажите его через --provider, например --provider {id}"
    ),
    "{name} is a paid provider: every call is billed.": (
        "{name} — платный провайдер: каждый вызов оплачивается."
    ),
    "{name} is billed per use; DESMAN uses it only with `desman run --provider {id}`.": (
        "{name} оплачивается за использование; DESMAN использует его только с "
        "`desman run --provider {id}`."
    ),
    "paid": "платный",
    "Paid providers are not tested in the prototype: they will be tested outside it.": (
        "Платные провайдеры в прототипе не тестировались: их проверят уже вне прототипа."
    ),
    "unknown provider {name!r} (known: {known})": "неизвестный провайдер {name!r} (известны: {known})",
    # ── config / keys / usage ────────────────────────────────────────────────
    "config {error}\nFix the file, or run `desman config reset` to start over.": (
        "настройки: {error}\nИсправьте файл или начните заново: `desman config reset`."
    ),
    "config {error}; run `desman config` again": "настройки: {error}; запустите `desman config` ещё раз",
    "Configuration cancelled; nothing was saved.": "Настройка отменена, ничего не сохранено.",
    "Config": "Настройки",
    "Language": "Язык",
    "Permissions": "Разрешения",
    "none": "нет",
    "Not configured yet: run `desman config`.": "Ещё не настроено: запустите `desman config`.",
    "WARNING: admin_rights is on: agents may act with elevated privileges.": (
        "ВНИМАНИЕ: admin_rights включён: агенты могут действовать с повышенными правами."
    ),
    "provider": "провайдер",
    "status": "статус",
    "keys": "ключи",
    "settings": "настройки",
    "Providers": "Провайдеры",
    "local": "локальный",
    "ready": "готов",
    "not set": "не задано",
    "unknown key {env}={values} (no provider uses this name)": (
        "неизвестный ключ {env}={values} (это имя не использует ни один провайдер)"
    ),
    "unknown setting {env}={value}": "неизвестная настройка {env}={value}",
    "No configuration at {path}; nothing to delete.": "Настроек в {path} нет, удалять нечего.",
    "Delete {path} and the {n} API key in it?": (
        "Удалить {path} и {n} API-ключ в нём?",
        "Удалить {path} и {n} API-ключа в нём?",
        "Удалить {path} и {n} API-ключей в нём?",
    ),
    "Delete {path} (it does not load)?": "Удалить {path} (файл не читается)?",
    "Kept.": "Оставлено.",
    "Deleted {path}. Run `desman config` to set DESMAN up again.": (
        "{path} удалён. Чтобы настроить DESMAN заново, запустите `desman config`."
    ),
    "It was a link: only the link is gone, {target} is left as it was.": (
        "Это была ссылка: удалена только она, {target} остался как был."
    ),
    "key": "ключ",
    "label": "метка",
    "calls 24h": "вызовы 24ч",
    "tokens 24h": "токены 24ч",
    "API keys": "API-ключи",
    "{n} key from {with_keys} of {providers} providers. `desman keys check` tests them "
    "(no tokens spent), `desman usage` shows totals.": (
        "{n} ключ у {with_keys} из {providers} провайдеров. `desman keys check` проверит их "
        "(без траты токенов), `desman usage` покажет расход.",
        "{n} ключа у {with_keys} из {providers} провайдеров. `desman keys check` проверит их "
        "(без траты токенов), `desman usage` покажет расход.",
        "{n} ключей у {with_keys} из {providers} провайдеров. `desman keys check` проверит их "
        "(без траты токенов), `desman usage` покажет расход.",
    ),
    "no keys to check: add one with `desman keys add PROVIDER`": (
        "нечего проверять: добавьте ключ через `desman keys add PROVIDER`"
    ),
    "{name} is not configured": "{name} не настроен",
    "models": "модели",
    "default model": "модель по умолчанию",
    "ok": "работает",
    "failed ({status})": "ошибка ({status})",
    "no reply": "нет ответа",
    "(not listed)": "(нет в списке)",
    "{name} limits and quotas: {url}": "Лимиты и квоты {name}: {url}",
    "No model calls yet.": "Вызовов моделей ещё не было.",
    "No model calls in the last {since}.": "За последние {since} вызовов моделей не было.",
    "Usage (all time)": "Расход (за всё время)",
    "Usage (last {since})": "Расход (за последние {since})",
    "calls": "вызовы",
    "failed": "ошибки",
    "limited": "лимиты",
    "tokens in": "вх. токены",
    "tokens out": "вых. токены",
    "last call": "последний",
    "removed ({id}…)": "удалён ({id}…)",
    '"removed": calls in this period went through a key that is no longer in the config '
    "(deleted or replaced). The characters in brackets start its fingerprint.": (
        "«Удалён»: вызовы за этот период шли через ключ, которого уже нет в настройках "
        "(его удалили или заменили). В скобках — начало его отпечатка."
    ),
    "limit": "лимит",
    "left": "осталось",
    "resets in": "сброс через",
    "Limits reported with the latest call": "Лимиты из последнего ответа провайдера",
    "{key}: the limit is 0, so this account has no access to the API yet (a free plan may "
    "need activating on the provider's site). DESMAN sets such a key aside and tries the "
    "provider's other keys.": (
        "{key}: лимит равен 0, у этого аккаунта пока нет доступа к API (бесплатный план, "
        "возможно, нужно активировать на сайте провайдера). DESMAN откладывает такой ключ и "
        "пробует другие ключи провайдера."
    ),
    "Grey rows come from an earlier call: later calls with that key sent no limits.": (
        "Серые строки — из более раннего вызова: следующие вызовы с этим ключом лимитов не "
        "присылали."
    ),
    "use e.g. 30m, 24h, 7d or all": "например 30m, 24h, 7d или all",
    "{name} is local and has no API key": "{name} — локальный провайдер, API-ключа у него нет",
    "Get a key: {url}": "Получить ключ: {url}",
    "API key for {name} ({env}), hidden": "API-ключ для {name} ({env}), ввод скрыт",
    "no key given": "ключ не введён",
    "{name} already has this key; nothing changed.": "У {name} уже есть этот ключ, ничего не изменено.",
    "Added {key} to {name} (key #{number}, {number} in total).": (
        "Ключ {key} добавлен к {name} (ключ №{number}, всего {number})."
    ),
    "{name} also needs {label} ({env}): set it in `desman config`.": (
        "{name} нужен ещё {label} ({env}): задайте в `desman config`."
    ),
    "give a key NUMBER or --all": "укажите номер ключа NUMBER или --all",
    "{name} has no keys": "у {name} нет ключей",
    "{name} has {n} key: pick 1": (
        "у {name} {n} ключ: выберите 1",
        "у {name} {n} ключа: выберите от 1 до {n}",
        "у {name} {n} ключей: выберите от 1 до {n}",
    ),
    "Removed {n} key from {name}; {left} left.": (
        "Удалён {n} ключ у {name}, осталось {left}.",
        "Удалено {n} ключа у {name}, осталось {left}.",
        "Удалено {n} ключей у {name}, осталось {left}.",
    ),
    "cannot read {file}: {error}": "не удаётся прочитать {file}: {error}",
    "no keys found in {file}": "в {file} нет ключей",
    "Imported from {file}": "Импорт из {file}",
    "added": "добавлено",
    "already saved": "уже было",
    "skipped names no provider uses: {names}": (
        "пропущены имена, которые не использует ни один провайдер: {names}"
    ),
    "{n} key added, {already} already saved.": (
        "Добавлен {n} ключ, {already} уже были сохранены.",
        "Добавлено {n} ключа, {already} уже были сохранены.",
        "Добавлено {n} ключей, {already} уже были сохранены.",
    ),
    "The keys are now in {path}; {file} can be deleted.": (
        "Теперь ключи хранятся в {path}, файл {file} можно удалить."
    ),
    # ── Bare `desman` ────────────────────────────────────────────────────────
    "Decentralized Self-Evolving Modular Agents Network": (
        "Децентрализованная самоэволюционирующая модульная сеть агентов"
    ),
    "{n} provider ready": (
        "готов {n} провайдер",
        "готово {n} провайдера",
        "готово {n} провайдеров",
    ),
    "no provider ready: run `desman config`": "нет готовых провайдеров: `desman config`",
    'desman run "GOAL"': 'desman run "ЦЕЛЬ"',
    "plan and run a task": "спланировать и выполнить задачу",
    "tasks and their progress": "задачи и их ход",
    "what waits for your decision": "что ждёт вашего решения",
    "API keys per provider": "API-ключи по провайдерам",
    "providers, keys, permissions": "провайдеры, ключи, разрешения",
    "all commands": "все команды",
    "config: {path}": "настройки: {path}",
    # ── Prompts ──────────────────────────────────────────────────────────────
    "(↑/↓, Enter)": "(↑/↓, Enter)",
    "(Space toggles, Enter confirms)": "(Пробел — отметить, Enter — подтвердить)",
    "enter a number from 1 to {n}": "введите число от 1 до {n}",
    "enter y or n": "введите y или n (д или н)",
    # ── Wizard ───────────────────────────────────────────────────────────────
    "Free tier": "Бесплатные",
    "Paid: billed per use": "Платные: оплата за использование",
    "Local": "Локальные",
    "Paid provider": "Платный провайдер",
    "Every call to {name} is billed to your account. DESMAN never picks a paid provider by "
    "itself: it is used only with `desman run --provider {id}`. Set a spending limit in the "
    "provider's console.": (
        "Каждый вызов {name} оплачивается с вашего счёта. Сам DESMAN платного провайдера "
        "никогда не выбирает: он используется только с `desman run --provider {id}`. "
        "Задайте лимит расходов в консоли провайдера."
    ),
    "I understand that calls to {name} cost money": "Я понимаю, что вызовы {name} платные",
    "DESMAN(P) — Configuration": "DESMAN(P) — Настройка",
    "Model providers": "Провайдеры моделей",
    "Pick a provider to add keys or change settings": (
        "Выберите провайдера, чтобы добавить ключи или изменить настройки"
    ),
    "Several providers and several keys per provider are fine: pick one, then the next.": (
        "Можно настроить несколько провайдеров и несколько ключей у каждого: выберите "
        "одного, потом следующего."
    ),
    "Done — continue": "Готово — дальше",
    "{n} key": ("{n} ключ", "{n} ключа", "{n} ключей"),
    "no key": "нет ключа",
    "needs {what}": "нужен {what}",
    "Data privacy": "Приватность данных",
    "Cloud models receive your prompts, context and data: they leave your machine and are "
    "processed by the provider.": (
        "Облачные модели получают ваши промпты, контекст и данные: они покидают ваш "
        "компьютер и обрабатываются провайдером."
    ),
    "I understand my data will be processed by third parties": (
        "Я понимаю, что мои данные обработают третьи стороны"
    ),
    "API key for {name} ({env}), hidden; Enter to cancel": (
        "API-ключ для {name} ({env}), ввод скрыт; Enter — отмена"
    ),
    "Label for this key (optional, e.g. 'second account')": (
        "Метка ключа (необязательно, например «второй аккаунт»)"
    ),
    "Key added to {name}.": "Ключ добавлен к {name}.",
    "{name} already has this key.": "У {name} уже есть этот ключ.",
    "That label looks like a key, so it was left out: a label is a name.": (
        "Метка похожа на ключ, поэтому не сохранена: метка — это название."
    ),
    "Add a key": "Добавить ключ",
    "Remove a key": "Удалить ключ",
    "Set {label} ({env})": "Задать {label} ({env})",
    "Back": "Назад",
    "Which key?": "Какой ключ?",
    "Key removed from {name}.": "Ключ удалён у {name}.",
    "{label} for {name} ({env}); '-' clears it": "{label} для {name} ({env}); '-' — очистить",
    "Agent permissions": "Разрешения агентов",
    "This is prototype software and agents can act unexpectedly. Grant only what you accept "
    "the risks of. Everything is off unless you turn it on.": (
        "Это прототип, агенты могут действовать неожиданно. Давайте только то, риски чего "
        "принимаете. Всё выключено, пока вы не включите."
    ),
    "Allow agents to": "Разрешить агентам",
    "act with administrator / root privileges": "действовать с правами администратора / root",
    "touch files outside the DESMAN workspace": "работать с файлами вне рабочей папки DESMAN",
    "use the network beyond model providers": "выходить в сеть помимо провайдеров моделей",
    "run shell commands": "выполнять shell-команды",
    "admin_rights is on: agents may act with elevated privileges.": (
        "admin_rights включён: агенты могут действовать с повышенными правами."
    ),
    "shell_execution is on, but there is no sandbox to run commands in ({reason}): run_command stays off.": "shell_execution включён, но песочницы для запуска команд нет ({reason}): run_command остаётся выключенным.",
    "Configuration saved to {path}": "Настройки сохранены в {path}",
    'Next: `desman config show` to review, `desman run "..."` to start.': (
        'Дальше: `desman config show` — проверить, `desman run "..."` — запустить.'
    ),
    "No cloud key yet: cloud models will not work until you add one.": (
        "Облачных ключей пока нет: облачные модели не заработают, пока не добавите."
    ),
    "Show the interface languages, or switch to LANGUAGE (a code such as ru, or its name).": "Показать языки интерфейса или переключиться на LANGUAGE (код, например ru, или название).",
    "code": "код",
    "language": "язык",
    "Languages": "Языки",
    "Switch with `desman language CODE`.": "Переключить: `desman language КОД`.",
    "Language set to {name}.": "Язык интерфейса: {name}.",
    "unknown language {name!r} (known: {codes})": "неизвестный язык {name!r} (известны: {codes})",
    "interface language": "язык интерфейса",
    "(Y/N)": "(Y/N)",
    "Ask the model even when the same request already has an accepted reply (by default such a reply is reused for 7 days, which saves quota on a resume).": "Спрашивать модель, даже если на такой же запрос уже есть принятый ответ (по умолчанию он используется повторно 7 дней, это экономит квоту при resume).",
    "yes": "да",
    "no": "нет",
    "Create a task for GOAL (or from --spec) and work on it.\n\nThe model splits GOAL into subtasks; if it cannot, the goal itself is the only subtask\n(see --plan). Exit codes: 0 done, 3 waiting for the operator, 4 paused (provider limit),\n5 failed or cancelled, 6 still in progress elsewhere.": "Создать задачу для цели GOAL (или из --spec) и выполнить её.\n\nМодель разбивает GOAL на подзадачи; если не получилось, единственной подзадачей становится сама цель (см. --plan). Коды выхода: 0 — готово, 3 — ждёт оператора, 4 — пауза (лимит провайдера), 5 — failed или cancelled, 6 — ещё выполняется в другом месте.",
    "--plan is for GOAL: a --spec is already a plan": "--plan относится к GOAL: --spec уже и есть план",
    "How GOAL becomes subtasks: llm asks the model for a plan (the default; if it fails, the template is used), template makes the goal itself the only subtask (the default with --fake-llm).": "Как GOAL превращается в подзадачи: llm просит модель составить план (по умолчанию; если не вышло, берётся шаблон), template делает саму цель единственной подзадачей (по умолчанию с --fake-llm).",
    "Planning…": "Составляю план…",
    "The model did not make a plan, so the goal is the only subtask: {error}": "Модель не составила план, поэтому единственная подзадача — сама цель: {error}",
    "Retries may also use {provider} · {model}": "Повторные попытки могут использовать и {provider} · {model}",
    "part of {seq}": "часть {seq}",
    "web search": "веб-поиск",
    "Web search (for the web_search tool)": "Веб-поиск (для инструмента web_search)",
    "SearXNG is a search engine you run yourself: it asks Google, Bing, DuckDuckGo and others for you, without keys or accounts. How to start one: {doc}": "SearXNG — поисковик, который вы запускаете сами: он сам опрашивает Google, Bing, DuckDuckGo и другие, без ключей и аккаунтов. Как запустить: {doc}",
    "SearXNG answers: web search works.": "SearXNG отвечает: веб-поиск работает.",
    "SearXNG at {url} cannot be used yet: {problem}": "SearXNG по адресу {url} пока недоступен: {problem}",
    "network_access is on, but no engine for the whole web is set up: web_search covers only Wikipedia, Stack Overflow, arXiv and GitHub. Add one in `desman config`.": "network_access включён, но поисковик для всего интернета не настроен: web_search ищет только в Википедии, Stack Overflow, arXiv и GitHub. Добавьте его в `desman config`.",
    # ── Chat (ROADMAP C.2) ──────────────────────────────────────────────────
    "the chat's commands": "команды чата",
    "start a new session": "начать новую сессию",
    "earlier sessions; pick one to continue it": "прошлые сессии; выберите, чтобы продолжить",
    "leave the chat (Ctrl+D does too)": "выйти из чата (или Ctrl+D)",
    'Saved in the session. Nobody answers here yet: the interlocutor comes in a later version. Meanwhile `desman run "GOAL"` runs a task.': 'Сохранено в сессии. Здесь пока никто не отвечает: собеседник появится в следующей версии. А пока задачу запускает `desman run "ЦЕЛЬ"`.',
    "New session. /help lists the chat's commands.": "Новая сессия. /help — команды чата.",
    "Continuing the session of {when} ({n} message). /new starts another, /help lists the chat's commands.": (
        "Продолжаем сессию от {when} ({n} сообщение). /new начинает другую, /help — команды чата.",
        "Продолжаем сессию от {when} ({n} сообщения). /new начинает другую, /help — команды чата.",
        "Продолжаем сессию от {when} ({n} сообщений). /new начинает другую, /help — команды чата.",
    ),
    "unknown command {name}: /help lists the chat's commands": "неизвестная команда {name}: /help — команды чата",
    "Anything else you write is a message.": "Всё остальное — сообщение.",
    "This session is new already.": "Эта сессия и так новая.",
    "New session: it starts with your next message.": "Новая сессия: она начнётся с вашего следующего сообщения.",
    "No sessions yet: the first starts with your first message.": "Сессий пока нет: первая начнётся с вашего первого сообщения.",
    "{n} message": (
        "{n} сообщение",
        "{n} сообщения",
        "{n} сообщений",
    ),
    "current": "текущая",
    "the new session": "новая сессия",
    "Which session?": "Какая сессия?",
    "Switched to the session of {when}.": "Переключились на сессию от {when}.",
    "earlier sessions…": "более ранние сессии…",
}

CLICK: dict[str, Forms] = {
    # Help pages
    "Usage:": "Использование:",
    "Options": "Опции",
    "Commands": "Команды",
    "Show this message and exit.": "Показать эту справку и выйти.",
    "Show the version and exit.": "Показать версию и выйти.",
    "%(prog)s, version %(version)s": "%(prog)s, версия %(version)s",
    "default: {default}": "по умолчанию: {default}",
    "required": "обязательно",
    "env var: {var}": "переменная окружения: {var}",
    "deprecated": "устарело",
    # Errors
    "Error: {message}": "Ошибка: {message}",
    "Try '{command} {option}' for help.": "Справка: '{command} {option}'.",
    "No such command {name!r}.": "Нет такой команды {name!r}.",
    "No such option {name!r}.": "Нет такой опции {name!r}.",
    "Did you mean {possibility}?": (
        "Может быть, {possibility}?",
        "(Может быть, одна из: {possibilities}?)",
        "(Может быть, одна из: {possibilities}?)",
    ),
    "Missing argument": "Не хватает аргумента",
    "Missing option": "Не хватает опции",
    "Missing parameter": "Не хватает параметра",
    "Missing {param_type}": "Не хватает: {param_type}",
    "Missing parameter: {param_name}": "Не хватает параметра: {param_name}",
    "Missing command.": "Не указана команда.",
    "Invalid value: {message}": "Недопустимое значение: {message}",
    "Invalid value for {param_hint}: {message}": "Недопустимое значение {param_hint}: {message}",
    "Got unexpected extra argument ({args})": (
        "Лишний аргумент ({args})",
        "Лишние аргументы ({args})",
        "Лишние аргументы ({args})",
    ),
    "Option {name!r} requires an argument.": (
        "Опции {name!r} нужен аргумент.",
        "Опции {name!r} нужно {nargs} аргумента.",
        "Опции {name!r} нужно {nargs} аргументов.",
    ),
    "Option {name!r} does not take a value.": "Опция {name!r} не принимает значения.",
    "{value!r} is not {choice}.": (
        "{value!r} — не {choice}.",
        "{value!r} — не одно из: {choices}.",
        "{value!r} — не одно из: {choices}.",
    ),
    "{value!r} is not a valid {number_type}.": "{value!r} — недопустимое {number_type}.",
    "{value} is not in the range {range}.": "{value} не входит в диапазон {range}.",
    "{name} {filename!r} does not exist.": "{name} {filename!r} не существует.",
    "{name} {filename!r} is a directory.": "{name} {filename!r} — это каталог.",
    "{name} {filename!r} is a file.": "{name} {filename!r} — это файл.",
    "{name} {filename!r} is not readable.": "{name} {filename!r} нельзя прочитать.",
    "{name} {filename!r} is not writable.": "{name} {filename!r} нельзя записать.",
    "file": "файл",
    "directory": "каталог",
    "path": "путь",
    # Prompts
    "Aborted!": "Прервано!",
    "Error: invalid input": "Ошибка: неверный ввод",
    "Repeat for confirmation": "Повторите для подтверждения",
    "Error: The two entered values do not match.": "Ошибка: введённые значения не совпадают.",
    # Errors
    "No such option: {name}": "Нет такой опции: {name}",  # Click < 8.4
    "Choose from:\n\t{choices}": "Выберите из:\n\t{choices}",
    "Positional arguments": "Позиционные аргументы",
    "(dynamic)": "(вычисляется)",
    "unknown error": "неизвестная ошибка",
    "Could not open file {filename!r}: {message}": "Не удалось открыть файл {filename!r}: {message}",
    "Argument {name!r} takes {nargs} values.": "Аргумент {name!r} принимает {nargs} значения.",
    "Takes {nargs} values but 1 was given.": (
        "Принимает {nargs} значения, а передано 1.",
        "Принимает {nargs} значения, а передано {len}.",
        "Принимает {nargs} значения, а передано {len}.",
    ),
    "{len_type} values are required, but {len_value} was given.": (
        "Нужно значений: {len_type}, а передано {len_value}.",
        "Нужно значений: {len_type}, а передано {len_value}.",
        "Нужно значений: {len_type}, а передано {len_value}.",
    ),
    "{value!r} does not match the format {format}.": (
        "{value!r} не соответствует формату {format}.",
        "{value!r} не соответствует форматам {formats}.",
        "{value!r} не соответствует форматам {formats}.",
    ),
    "{value!r} is not a valid boolean.": "{value!r} — не логическое значение.",
    "{value!r} is not a valid boolean. Recognized values: {states}": (
        "{value!r} — не логическое значение. Допустимые значения: {states}"
    ),
    "{value!r} is not a valid UUID.": "{value!r} — недопустимый UUID.",
    "{name} {filename!r} is not executable.": "{name} {filename!r} нельзя выполнить.",
    "Invalid start character for option ({option})": (
        "Недопустимый первый символ опции ({option})"
    ),
    # Deprecation notices
    "(Deprecated) {text}": "(Устарело) {text}",
    "{text} {deprecated_message}": "{text} {deprecated_message}",
    "DeprecationWarning: The command {name!r} is deprecated.": (
        "DeprecationWarning: команда {name!r} устарела."
    ),
    "DeprecationWarning: The command {name!r} is deprecated.{extra_message}": (
        "DeprecationWarning: команда {name!r} устарела.{extra_message}"
    ),
    "DeprecationWarning: The {param_type} {name!r} is deprecated.{extra_message}": (
        "DeprecationWarning: {param_type} {name!r} устарел(а).{extra_message}"
    ),
    # Prompts (Click < 8.4 spellings first)
    "Error: The value you entered was invalid.": "Ошибка: введено недопустимое значение.",
    "Error: {e.message}": "Ошибка: {e.message}",
    "Press any key to continue...": "Нажмите любую клавишу, чтобы продолжить...",
    "Do you want to continue?": "Продолжить?",
    "Confirm the action without prompting.": "Подтвердить действие без вопроса.",
    # Mistakes in the program's own command definitions: a developer sees these, not the
    # operator, but a Russian page should not switch language on them either.
    "Value must be an iterable.": "Значение должно быть итерируемым.",
    "Could not determine name for option with declarations {decls!r}": (
        "Не удалось определить имя опции с объявлениями {decls!r}"
    ),
    "No options defined but a name was passed ({name}). Did you mean to declare an argument "
    "instead? Did you mean to pass '--{name}'?": (
        "Опции не объявлены, но передано имя ({name}). Может быть, нужен аргумент? Или '--{name}'?"
    ),
    "Arguments take exactly one parameter declaration, got {length}: {decls}.": (
        "У аргумента должно быть ровно одно объявление, получено {length}: {decls}."
    ),
    "Name '{name}' defined twice": "Имя '{name}' объявлено дважды",
    "Boolean option {decl!r} cannot use the same flag for true/false.": (
        "Логическая опция {decl!r} не может использовать один флаг для true и false."
    ),
    "Choice({choices})": "Choice({choices})",
    "Unknown color {colour!r}": "Неизвестный цвет {colour!r}",
    "Unknown standard stream '{name}'": "Неизвестный стандартный поток '{name}'",
}

CATALOG = Catalog("Русский", plural, MESSAGES, CLICK, forms=3)

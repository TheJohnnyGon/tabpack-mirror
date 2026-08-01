# Папка lib/ — Базовая библиотека

## Назначение

Фундаментальная библиотека проекта, предоставляющая утилиты для работы с данными, экспериментами, окружением и типами.

## Структура

```
lib/
├── __init__.py
├── data.py           # Загрузка и преобразование данных (см. DATA_MODULE.md)
├── types.py          # Базовые типы
├── env.py            # Окружение и пути
├── util.py           # Утилиты
├── experiment.py     # Управление экспериментами
├── deep.py           # Утилиты для нейросетей
├── metrics.py        # Метрики (NumPy)
└── parallel.py       # Параллельное выполнение
```

---

## [`lib/types.py`](lib/types.py:1) — Базовые типы

### Type Aliases

| Тип | Описание |
|-----|----------|
| `KWArgs` | `dict[str, Any]` |
| `JSONDict` | `dict[str, Any]` (JSON-сериализуемый) |
| `AMPDType` | `Literal['bfloat16', 'float16']` |
| `DataKey` | `str` — 'x_num', 'x_bin', 'x_cat', 'y' |
| `PartKey` | `str` — 'train', 'val', 'test' |

### [`TaskType`](lib/types.py:12)

```python
class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
```

### [`PredictionType`](lib/types.py:18)

```python
class PredictionType(enum.Enum):
    LABELS = 'labels'
    PROBS = 'probs'
    LOGITS = 'logits'
```

---

## [`lib/env.py`](lib/env.py:1) — Окружение

**Важно:** Модуль импортирует ТОЛЬКО стандартную библиотеку.

### Функции

| Функция | Возвращает | Описание |
|---------|------------|----------|
| [`get_project_dir()`](lib/env.py:10) | `Path` | Корень проекта (по pyproject.toml) |
| [`get_exp_dir()`](lib/env.py:37) | `Path` | `{project}/exp` |
| [`get_cache_dir()`](lib/env.py:41) | `Path` | `{project}/cache` |
| [`get_data_dir()`](lib/env.py:47) | `Path` | `{project}/data` |
| [`get_local_dir()`](lib/env.py:51) | `Path` | `{project}/local` |
| [`get_snapshot_dir()`](lib/env.py:55) | `Path \| None` | Из SNAPSHOT_PATH |
| [`get_tmp_output_dir()`](lib/env.py:60) | `Path \| None` | Из TMP_OUTPUT_PATH |
| [`is_local()`](lib/env.py:65) | `bool` | True если нет SNAPSHOT_PATH |

---

## [`lib/util.py`](lib/util.py:1) — Утилиты

### Типы и валидация

#### [`check_typed_dict()`](lib/util.py:64)
Проверяет что словарь соответствует TypedDict (required/optional keys).

#### [`dataclass_from_dict()`](lib/util.py:266)
Создаёт dataclass из dict с полной проверкой типов.

**Поддерживаемые типы:**
- Простые: None, bool, int, float, str, bytes
- Path, Enum, Literal
- Union, Optional
- Collections: tuple, list, set, dict, frozenset
- Вложенные dataclass

### Dict утилиты

#### [`flatten_dict()`](lib/util.py:304)
Сплющивает вложенный dict: `{'a': {'b': 1}}` → `{'a.b': 1}`

### Import утилиты

#### [`import_()`](lib/util.py:310)
Динамический импорт: `import_('bin.demo.main')` → функция main

#### [`get_function_full_name()`](lib/util.py:323)
Получает полное имя: `bin.model.main`

### Device и GPU

#### [`get_device()`](lib/util.py:355)
Автоопределение: cuda:0 → mps:0 → cpu

#### [`get_amp_dtype()`](lib/util.py:372)
Конвертирует 'bfloat16'/'float16' в torch.dtype

#### [`is_oom_exception()`](lib/util.py:391)
Проверяет является ли ошибка OOM

#### [`adjust_gpu_memory_usage()`](lib/util.py:404)
Декоратор: при OOM уменьшает batch_size в 2 раза и повторяет

### Инициализация

#### [`init()`](lib/util.py:473)
- Проверяет что cwd == project_dir
- [`configure_logging()`](lib/util.py:456) — настраивает loguru
- [`configure_torch()`](lib/util.py:463) — determinism, no tf32

---

## [`lib/experiment.py`](lib/experiment.py:1) — Управление экспериментами

### Файлы эксперимента

| Файл | Описание |
|------|----------|
| `config.toml` | Конфигурация |
| `report.json` | Отчёт с метриками |
| `summary.txt` | Текстовый summary |
| `checkpoint.pt` | Чекпоинт для resume |
| `predictions.npz` | Предсказания |

### IO функции

| Функция | Описание |
|---------|----------|
| [`load_config()`](lib/experiment.py:65) | Загрузить config.toml |
| [`load_report()`](lib/experiment.py:71) | Загрузить report.json |
| [`dump_report()`](lib/experiment.py:107) | Сохранить report.json |
| [`dump_checkpoint()`](lib/experiment.py:118) | Сохранить checkpoint.pt |
| [`load_checkpoint()`](lib/experiment.py:82) | Загрузить checkpoint.pt |

### Status функции

| Функция | Описание |
|---------|----------|
| [`is_fresh()`](lib/experiment.py:146) | Только config.toml |
| [`is_done()`](lib/experiment.py:156) | Есть report.json и нет _RUNNING |
| [`is_experiment()`](lib/experiment.py:138) | Есть config.toml |

### Actions

| Функция | Описание |
|---------|----------|
| [`create()`](lib/experiment.py:164) | Создать эксперимент |
| [`reset()`](lib/experiment.py:217) | Удалить все кроме config |
| [`finish()`](lib/experiment.py:205) | Сохранить report + summary + backup |
| [`copy()`](lib/experiment.py:230) | Копировать эксперимент |
| [`move()`](lib/experiment.py:236) | Переместить |
| [`remove()`](lib/experiment.py:242) | Удалить |
| [`duplicate()`](lib/experiment.py:254) | Копировать в другой проект |

### [`run()`](lib/experiment.py:256) — Запуск эксперимента

**Логика:**
1. Если fresh → запустить
2. Если --force → reset и запустить
3. Если --resume и не done → продолжить
4. Если done → вернуть None

### [`run_cli()`](lib/experiment.py:427) — CLI запуск

Парсер: `exp --force --resume`

### [`summarize()`](lib/experiment.py:523) — Текстовый отчёт

Форматирует report в читаемый текст.

---

## [`lib/deep.py`](lib/deep.py:1) — Утилиты для нейросетей

### Инициализация

#### [`init_rsqrt_uniform_()`](lib/deep.py:16)
Xavier uniform инициализация: `U(-1/√d, 1/√d)`

### Модули

#### [`OneHotEncoding`](lib/deep.py:25)
One-hot кодирование (та же реализация что и в nn.py)

### Статистика

#### [`get_n_parameters()`](lib/deep.py:60)
Количество обучаемых параметров

#### [`compute_parameter_statistics()`](lib/deep.py:65)
Вычисляет norm, gradnorm, gradratio для всех параметров

### Оптимизация

#### [`default_zero_weight_decay_condition()`](lib/deep.py:92)
Определяет какие параметры НЕ должны иметь weight_decay:
- bias
- BatchNorm1d, LayerNorm, InstanceNorm1d
- LinearEmbeddings, LinearReLUEmbeddings

#### [`make_parameter_groups()`](lib/deep.py:110)
Создаёт 3 группы:
1. Default (с weight_decay)
2. Zero weight_decay (bias, norm layers)
3. Custom groups (например, Muon)

---

## [`lib/metrics.py`](lib/metrics.py:1) — Метрики (NumPy)

### [`calculate_metrics()`](lib/metrics.py:34)

**Для регрессии:**
- rmse, mae, r2

**Для классификации:**
- classification_report (precision, recall, f1)
- cross-entropy
- roc-auc (для binclass)

### [`_get_labels_and_probs()`](lib/metrics.py:10)
Конвертирует prediction_type → labels + probs

---

## [`lib/parallel.py`](lib/parallel.py:1) — Параллельное выполнение

### Worker management

| Функция | Описание |
|---------|----------|
| [`get_worker_id()`](lib/parallel.py:39) | ID текущего воркера или None |
| [`lock()`](lib/parallel.py:44) | Lock для thread-safety |

### [`map()`](lib/parallel.py:116)

**Назначение:** Параллельное выполнение функции.

**Параметры:**
- `fn` — функция
- `kwargs_list` — список kwargs
- `n_workers` — количество процессов
- `n_gpus_per_worker` — GPU на воркера

**Особенности:**
- Использует 'spawn' метод (требуется для CUDA)
- Распределяет CUDA_VISIBLE_DEVICES между воркерами
- Thread-safe lock для общих ресурсов

---

## Зависимости между модулями lib/

```
env.py (нет зависимостей от lib)
    ↓
util.py → env.py, types.py
    ↓
experiment.py → env.py, util.py, types.py
    ↓
data.py → env.py, util.py, types.py, metrics.py
    ↓
deep.py → (внешние: rtdl_*)
    ↓
parallel.py → util.py, types.py
    ↓
metrics.py → types.py
```

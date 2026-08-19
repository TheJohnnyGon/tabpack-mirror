# Подробное описание `convert_polars.py`

## 1. Назначение

[`convert_polars.py`](convert/convert_polars.py) — скрипт для конвертации предварительно отфильтрованных TSV-дампов в формат датасета TabPack. Является основной точкой входа для подготовки данных из Yandex.Train (YT) в формат, пригодный для обучения нейросетевых моделей TabPack.

**Ключевая проблема:** CatBoost заточен под работу с YT-таблицами в специфическом формате, который не подходит для обучения TabPack. Скрипт решает задачу преобразования данных в бинарный NumPy-формат (.npy) с memmap-поддержкой для эффективной работы с большими датасетами (80-100 ГБ).

## 2. Эволюция скрипта

### 2.1 Исходная версия (pandas)

- Использовался pandas для чтения TSV
- Время обработки 80 ГБ: **~8 часов**
- Проблемы:
  - pandas не оптимизирован для больших файлов
  - Высокое потребление памяти (полная загрузка в RAM)
  - Отсутствие контроля над типами данных (лишние копии)

### 2.2 Текущая версия (polars)

- Переход на polars с batched reading
- Время обработки 80 ГБ: **~1 час**
- Ускорение: **~8x**
- Постоянное потребление памяти (не зависит от размера файла)

### 2.3 Хронология изменений

| Коммит | Изменение |
|--------|-----------|
| [`28bac17`](convert/convert_polars.py:1) | Изначальный преобразователь YT → NPY |
| [`c629030`](convert/convert_polars.py:1) | Переделан convert_bce |
| [`81a3fc2`](convert/convert_polars.py:1) | Добавлен convert_polars |
| [`d9e9b7f`](convert/convert_polars.py:1) | Логгирование в convert, убраны старые файлы |
| [`cbd3663`](convert/convert_polars.py:1) | Доработка convert |
| [`aef1370`](convert/convert_polars.py:1) | Вспомогательные функции для convert |
| [`aa97227`](convert/convert_polars.py:1) | Streaming mode для 50M+ строк |
| [`3c7cda8`](convert/convert_polars.py:1) | Fix pickle при параллельной обработке |
| [`0915d7b`](convert/convert_polars.py:1) | Оптимизация для датасетов 80-100 ГБ |
| [`77d2ac6`](convert/convert_polars.py:1) | Fix обработка батчей в load_tsv() |
| [`172c84d`](convert/convert_polars.py:1) | Original keys для джойна с серпами |

## 3. Архитектура

### 3.1 Входные данные

**Формат TSV:**
```
key \t label \t f_1 \t f_2 \t ... \t f_K
```

- **key** — строковый идентификатор (DocId/OID документа)
- **label** — числовая метка (Float64)
- **f_1...f_K** — числовые признаки (Float32)
- Без заголовка (headerless)
- Колонки предварительно отфильтрованы через YQL-запрос (генерируется [`generate_yql_query.py`](convert/generate_yql_query.py))

**Источники TSV:**
- `--train` — тренировочная выборка
- `--val` — валидационная выборка
- `--test` — тестовая выборка

### 3.2 Выходные данные

| Файл | Тип | Описание |
|------|-----|----------|
| `x_num.npy` | memmap float32 (N, K) | Числовые признаки |
| `y.npy` | int64/float32 (N,) | Метки (зависит от task_type) |
| `key.npy` | int64 (N,) | Polars-хеш ключа (для группировки) |
| `original_keys.tsv` | str (N,) | Оригинальные (нехешированные) ключи |
| `info.json` | JSON | Метаданные задачи |
| `splits/default/{train,val,test}.npy` | int32 | Индексы сплитов |

### 3.3 Pipeline обработки

```
┌─────────────────────────────────────────────────────────────┐
│                     Входные TSV файлы                        │
│  train.tsv  │  val.tsv  │  test.tsv                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Sniff (определение числа признаков)                │
│  - Чтение первой строки каждого файла                        │
│  - Проверка консистентности n_features                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Count rows (быстрый подсчёт строк)                 │
│  - Бинарный подсчёт '\n' в блоках по 16 МБ                   │
│  - Необходим для preallocation memmap                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Allocate outputs                                   │
│  - x_num.npy: np.memmap (N, K) float32                      │
│  - ys: np.empty(N) float64                                  │
│  - keys: np.empty(N) int64                                  │
│  - orig_keys: np.empty(N) object                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Batched conversion (основной цикл)                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  for each split (train/val/test):                   │    │
│  │    pl.read_csv_batched(batch_size=500_000)          │    │
│  │    for each batch:                                  │    │
│  │      - labels → ys[sl]                             │    │
│  │      - keys → polars hash → keys[sl]               │    │
│  │      - orig keys → orig_keys[sl]                   │    │
│  │      - features → x_num[sl, :]                     │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Save metadata                                      │
│  - y.npy (astype к target dtype)                            │
│  - key.npy                                                  │
│  - original_keys.tsv (по одному ключу на строку)            │
│  - info.json (task type + score metric)                     │
│  - splits/default/{train,val,test}.npy                      │
└─────────────────────────────────────────────────────────────┘
```

## 4. Детальный разбор функций

### 4.1 [`parse_args()`](convert/convert_polars.py:53)

**CLI-парсер:**

| Аргумент | Тип | Обязательный | Описание |
|----------|-----|-------------|----------|
| `--train` | Path | Нет* | Train split TSV |
| `--val` | Path | Нет* | Validation split TSV |
| `--test` | Path | Нет* | Test split TSV |
| `--out-dir` | Path | Да | Output dataset directory |
| `--task-type` | Choice | Нет | binclass/multiclass/regression/pairwise (default: pairwise) |
| `--n-workers` | int | Нет | Число workers для polars (default: all CPUs) |

\* Хотя бы один из --train/--val/--test обязателен.

**Поддерживаемые типы задач:**

```python
TASK_SCORES = {
    "binclass": "accuracy",
    "multiclass": "accuracy",
    "regression": "rmse",
    "pairwise": "pair_accuracy",
}
```

### 4.2 [`sniff(path)`](convert/convert_polars.py:72)

Определяет число признаков из первой строки файла.

**Алгоритм:**
1. Открывает файл в текстовом режиме
2. Пропускает пустые строки
3. Считает число табов в первой непустой строке
4. Возвращает `len(line.split("\t")) - 2` (минус key и label)

**Пример:**
```
"doc_123\t1.0\t0.5\t0.3\t0.8"  →  split = 5 элементов  →  n_features = 3
```

### 4.3 [`load_tsv()`](convert/convert_polars.py:85)

**Основная функция загрузки.** Batch-based loader с постоянным потреблением памяти.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `path` | Path | Путь к TSV-файлу |
| `row0` | int | Начальный индекс строки в глобальных массивах |
| `x_num` | np.memmap | Массив признаков (предаллоцирован) |
| `ys` | np.ndarray | Массив меток |
| `keys` | np.ndarray | Массив хешей ключей |
| `orig_keys` | np.ndarray | Массив оригинальных ключей |
| `n_features` | int | Число признаков |
| `n_workers` | int | Число потоков polars |

**Schema определения:**

```python
# Polars называет колонки без заголовка как column_1, column_2, ... (1-based)
key_col = "column_1"          # ключ (String)
label_col = "column_2"        # метка (Float64)
feat_cols = ["column_3", ..., "column_{K+2}"]  # признаки (Float32)

schema = {
    key_col: pl.String,
    label_col: pl.Float64,
    feat_cols[i]: pl.Float32 for all i
}
```

**Batched reading:**

```python
reader = pl.read_csv_batched(
    str(path),
    separator="\t",
    has_header=False,
    try_parse_dates=False,       # Важно! Иначе polars пытается парсить даты
    null_values=list(NA_STRINGS), # {"", "nan", "null", "none", "\\n", "NaN", "NULL"}
    schema_overrides=schema,      # Явные типы (без лишних кастов)
    n_threads=n_workers,          # Параллелизм внутри файла
    batch_size=BATCH_SIZE,        # 500_000 строк
)
```

**Обработка каждого батча:**

```python
while True:
    batches = reader.next_batches(BATCH_SIZE)
    if batches is None:
        break
    
    batch = pl.concat(batches)  # Важно! next_batches возвращает list
    n = len(batch)
    sl = slice(row, row + n)
    
    # Labels — прямая конвертация (уже Float64 из schema)
    labels = batch[label_col].to_numpy()
    if np.isnan(labels).any():
        sys.exit(f"error: label is null/missing")
    ys[sl] = labels
    
    # Keys — polars hash (uint64 → int64 view)
    keys[sl] = batch[key_col].hash().to_numpy().view(np.int64)
    
    # Original keys — строки, тот же порядок что и x_num/y/key
    orig_keys[sl] = batch[key_col].cast(pl.String).to_numpy()
    
    # Features — прямая конвертация (уже Float32, нет .astype() копии!)
    x_num[sl, :] = batch.select(feat_cols).to_numpy()
    
    row += n
```

**Ключевые оптимизации:**

1. **Нет лишних `.astype()` копий:** колонки уже Float32/Float64 из schema_overrides
2. **Прямая запись в memmap:** `x_num[sl, :] = ...` без промежуточных переменных
3. **Single pass:** нет предварительного подсчёта строк через polars
4. **Constant memory:** батчи по 500k строк, память не растёт с размером файла

**Логгирование:**

```
  batch 1: 500,000 rows
  batch 2: 1,000,000 rows
  ...
  Total: 120,000,000 rows in 240 batches, 45.67s (2,627,569 rows/s)
```

### 4.4 [`main()`](convert/convert_polars.py:173)

**Основной цикл:**

#### Phase 1: Sniff
```python
n_feats_seen = set()
for name, path in splits:
    nf = sniff(path)
    n_feats_seen.add(nf)
if len(n_feats_seen) != 1:
    sys.exit(f"error: inconsistent feature counts: {n_feats_seen}")
```

#### Phase 2: Count rows
```python
counts = {}
for name, path in splits:
    n = 0
    with path.open("rb") as f:
        while block := f.read(1 << 24):  # 16 MB chunks
            n += block.count(b"\n")
    counts[name] = n
```

Быстрый бинарный подсчёт строк (не парсит содержимое, только считает `\n`).

#### Phase 3: Allocate
```python
x_num = np.lib.format.open_memmap(
    args.out_dir / "x_num.npy", mode="w+",
    dtype=np.float32, shape=(n_total, n_features),
)
ys = np.empty(n_total, dtype=np.float64)
keys = np.empty(n_total, dtype=np.int64)
orig_keys = np.empty(n_total, dtype=object)
```

#### Phase 4: Convert
```python
offsets = {}
row = 0
for name, path in tqdm(splits, desc="Files"):
    n = load_tsv(path, row, x_num, ys, keys, orig_keys, n_features, args.n_workers)
    if n != counts[name]:
        sys.exit(f"error: parsed {n} rows, counted {counts[name]}")
    offsets[name] = (row, row + n)
    row += n

x_num.flush()  # Фиксируем memmap на диск
```

#### Phase 5: Save metadata
```python
# Labels с правильным dtype
y_dtype = np.float32 if args.task_type == "regression" else np.int64
y = ys.astype(y_dtype)
np.save(args.out_dir / "y.npy", y)
np.save(args.out_dir / "key.npy", keys)

# Original keys — TSV для джойна с серпами
with orig_keys_path.open("w", encoding="utf-8") as f:
    for k in orig_keys:
        f.write("" if k is None else str(k))
        f.write("\n")

# Info.json
info = {"task": {"type": args.task_type, "score": TASK_SCORES[args.task_type]}}
(args.out_dir / "info.json").write_text(json.dumps(info, indent=4) + "\n")

# Splits
split_dir = args.out_dir / "splits" / "default"
split_dir.mkdir(parents=True, exist_ok=True)
for name, (a, b) in offsets.items():
    np.save(split_dir / f"{name}.npy", np.arange(a, b, dtype=np.int32))
```

## 5. Оптимизации

### 5.1 Memory optimizations

| Оптимизация | Описание | Эффект |
|-------------|----------|--------|
| **memmap для x_num** | Признаки хранятся на диске, не в RAM | RAM не растёт с N |
| **Batched reading** | 500k строк за раз | Constant memory |
| **schema_overrides** | Явные типы при чтении | Нет лишних .astype() копий |
| **Direct memmap writes** | `x_num[sl, :] = batch.to_numpy()` | Нет промежуточных переменных |
| **No count_rows pre-scan** | Быстрый бинарный подсчёт `\n` | Один проход по файлу |

### 5.2 Performance optimizations

| Оптимизация | Описание | Эффект |
|-------------|----------|--------|
| **polars вместо pandas** | Polars использует Rust под капотом | ~8x ускорение |
| **n_threads=n_workers** | Полная загрузка всех CPU | Максимальный параллелизм |
| **try_parse_dates=False** | Отключён парсинг дат | Ускорение чтения |
| **BATCH_SIZE=500_000** | Оптимальный размер батча | Баланс между памятью и оверхедом |

### 5.3 Correctness optimizations

| Оптимизация | Описание |
|-------------|----------|
| **NA_STRINGS handling** | Корректная обработка null/nan/none/NULL |
| **Label validation** | Проверка на NaN в метках с точным номером строки |
| **Row count validation** | Проверка совпадения подсчитанных и распарсенных строк |
| **Feature count consistency** | Проверка консистентности числа признаков между файлами |

## 6. Original Keys механизм

### 6.1 Проблема

Раньше в датасете сохранялся только polars-хеш ключа (`key.npy`, int64), который:
- Необратим (нельзя восстановить оригинальный ключ из хеша)
- Не совпадает с OID из target_table.json
- Невозможно заджойнить результаты инференса с метаданными документов

### 6.2 Решение

Рядом с `key.npy` сохраняется `original_keys.tsv`:
- По одному ключу на строку
- Строка i в TSV соответствует строке i в `x_num.npy`
- Полный порядок датасета до применения сплита

### 6.3 Использование в инференсе

```bash
uv run python infer/infer.py --parts test --metrics \
    --metrics-output-dir infer/out
```

Инференс:
1. Читает `original_keys.tsv`
2. Выравнивает ключи с партом через индексы сплита
3. Для каждого объекта берёт RawFormulaVal (усреднённый по ансамблю предикт)
4. Пишет TSV `<original_key>\t<RawFormulaVal>`

Результат можно заджойнить с метаданными документов для сборки серпа.

## 7. Интеграция с generate_yql_query.py

### 7.1 Pipeline подготовки данных

```
YT Table (сырые данные)
       │
       ▼
  YQL Query (generate_yql_query.py)
  - Фильтрация ignored-features на уровне SQL
  - Выбор нужных колонок (CD, features)
       │
       ▼
  TSV dump (pre-filtered)
       │
       ▼
  convert_polars.py
  - Конвертация в .npy формат
       │
       ▼
  TabPack dataset (x_num.npy, y.npy, key.npy, ...)
```

### 7.2 Преимущества

1. **ignored-features убираются в SQL:** не нужно загружать и потом отбрасывать лишние колонки
2. **Меньший размер TSV:** только нужные колонки
3. **Быстрее конвертация:** меньше данных для обработки

## 8. Использование

### 8.1 Базовый пример

```bash
python convert/convert_polars.py \
    --train data/train.tsv \
    --val data/val.tsv \
    --test data/test.tsv \
    --out-dir data/peru \
    --task-type pairwise \
    --n-workers 8
```

### 8.2 Только train

```bash
python convert/convert_polars.py \
    --train data/train.tsv \
    --out-dir data/peru \
    --task-type binclass
```

### 8.3 Ожидаемый вывод

```
Features: 128, workers: 8
Rows: {'train': 120000000, 'val': 10000000, 'test': 5000000} (total 135000000)
Loading train (120,000,000 rows)...
  batch 1: 500,000 rows
  batch 2: 1,000,000 rows
  ...
  Total: 120,000,000 rows in 240 batches, 45.67s (2,627,569 rows/s)
  train: rows [0, 120000000)
Loading val (10,000,000 rows)...
  ...
  val: rows [120000000, 130000000)
Loading test (5,000,000 rows)...
  ...
  test: rows [130000000, 135000000)

Saved to data/peru:
  x_num.npy (135000000, 128) float32
  y.npy (135000000,) int64
  key.npy (135000000,) int64
  original_keys.tsv (135000000,) str
  info.json
  splits/default/{train, val, test}.npy
  label distribution: {0: 67500000, 1: 67500000}

Done.
```

## 9. Ограничения и известные проблемы

### 9.1 Ограничения

| Ограничение | Описание |
|-------------|----------|
| **Все признаки numeric** | Скрипт не обрабатывает категориальные признаки |
| **Без заголовка** | TSV файлы не должны иметь header |
| **Консистентные признаки** | Все файлы должны иметь одинаковое число признаков |
| **Нет missing labels** | Метки не могут быть null/NaN |

### 9.2 Известные проблемы

| Проблема | Решение |
|----------|---------|
| `TypeError: list indices must be integers or slices, not str` | `reader.next_batches()` возвращает list — нужен `pl.concat(batches)` |
| Pickle error при параллельной обработке | Исправлено в коммите 3c7cda8 |

## 10. Сравнение с альтернативами

### 10.1 Pandas vs Polars

| Метрика | Pandas | Polars |
|---------|--------|--------|
| Время (80 ГБ) | ~8 часов | ~1 час |
| Пиковая память | Высокая (полная загрузка) | Constant (батчи) |
| Параллелизм | Ограниченный | Полная загрузка CPU |
| Типы данных | Implicit casting | Explicit schema |

### 10.2 Почему memmap для x_num?

- **x_num.npy** — самый большой файл (N × K × 4 байта)
- При N=100M, K=128: 100M × 128 × 4 = 50 ГБ
- Memmap позволяет работать с файлом на диске, не загружая всё в RAM
- При доступе к батчу ОС кэширует нужные страницы автоматически

## 11. Структура выходного датасета

```
data/peru/
├── x_num.npy              # memmap float32 (N, K) — признаки
├── y.npy                  # int64/float32 (N,) — метки
├── key.npy                # int64 (N,) — хеши ключей
├── original_keys.tsv      # str (N,) — оригинальные ключи
├── info.json              # {"task": {"type": "pairwise", "score": "pair_accuracy"}}
└── splits/
    └── default/
        ├── train.npy      # int32 — индексы train сплита
        ├── val.npy        # int32 — индексы val сплита
        └── test.npy       # int32 — индексы test сплита
```

**Важно:** индексы в сплитах — это contiguous ranges в оригинальном порядке строк. Группы (по ключу) остаются contiguous, что готовит данные для будущих pairlogit/group задач.

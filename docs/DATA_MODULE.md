# Модуль data.py — Работа с данными

## Назначение

Модуль отвечает за загрузку, преобразование и стандартизацию табличных данных для обучения.

---

## Типы данных

### Константы dtype

| Константа | Тип | Назначение |
|-----------|-----|------------|
| [`_X_NUM_DTYPE`](lib/data.py:27) | `np.float32` | Числовые признаки |
| [`_X_BIN_DTYPE`](lib/data.py:28) | `np.float32` | Бинарные признаки |
| [`_X_CAT_INT_DTYPE`](lib/data.py:29) | `np.int64` | Категориальные (интегральные) |
| [`_X_CAT_STR_DTYPE`](lib/data.py:36) | `np.str_` | Категориальные (строковые) |
| [`_Y_REG_DTYPE`](lib/data.py:37) | `np.float32` | Цели регрессии |
| [`_Y_CLF_DTYPE`](lib/data.py:38) | `np.int64` | Цели классификации |
| [`_SPLIT_DTYPE`](lib/data.py:39) | `np.int32` | Индексы сплитов |

---

## DataPreprocessor

### [`DataPreprocessor`](lib/data.py:54)

**Назначение:** Препроцессор с паттерном fit/transform для воспроизводимости.

**Важно:** Все трансформеры и индексы признаков хранятся как атрибуты экземпляра класса, а не как глобальные переменные. Это обеспечивает потокобезопасность и позволяет сохранять препроцессор в `model.pt` для инференса.

**Параметры конфига:**
- `seed` — случайное зерно
- `extract_bin_from_num` — извлекать бинарные из числовых
- `skip_bin_encoder` — пропускать OrdinalEncoder для бинарных
- `num_policy` — 'standard' или 'noisy-quantile'
- `num_memory_efficient` — columnwise трансформация
- `bin_policy` — 'convert-to-cat'
- `cat_policy` — 'ordinal' или 'one-hot'

**Атрибуты экземпляра:**
- `feature_indices_num`, `feature_indices_bin`, `feature_indices_cat` — индексы признаков
- `transformer_num` — StandardScaler или QuantileTransformer
- `transformer_bin_ordinal` — OrdinalEncoder для бинарных
- `transformer_cat_ordinal` — OrdinalEncoder для категориальных
- `transformer_cat_onehot` — OneHotEncoder для категориальных (если cat_policy='one-hot')

### Методы

#### [`fit()`](lib/data.py:98)

Обучает трансформеры на train данных:

1. **Извлечение бинарных** — определяет столбцы с 2 уникальными значениями
2. **Num трансформер** — StandardScaler или QuantileTransformer на train
3. **Cat OrdinalEncoder** — на train данных
4. **Cat OneHotEncoder** — если cat_policy='one-hot'

**Важно:** Fit должен вызываться только на train данных, чтобы избежать data leakage!

#### [`transform()`](lib/data.py:225)

Применяет обученные трансформеры:

1. Разделяет x_num на x_num и x_bin по сохранённым индексам
2. Применяет num трансформер ко всем частям
3. Удаляет константные столбцы
4. Применяет cat трансформеры
5. Обрабатывает неизвестные категории в val/test

**Важно:** Transform может вызываться на train/val/test данных после fit.

#### [`fit_transform()`](lib/data.py:366)

Комбинация fit + transform. Вызывается в `main()` для обучения:

```python
preprocessor = DataPreprocessor(data_config_with_seed)
dataset = preprocessor.fit_transform(dataset)
```

**При инференсе:**

```python
# Загрузить сохранённый preprocessor из model.pt
preprocessor = artifact['preprocessor']
# Применить transform БЕЗ fit
new_dataset = preprocessor.transform(new_dataset)
```

---

## Загрузка данных

### [`load_data()`](lib/data.py:492)

**Назначение:** Загрузить все .npy файлы и применить сплит.

**Процесс:**
1. Загружает все .npy файлы из директории
2. Загружает сплит по split_id
3. Применяет сплит ко всем данным
4. Проверяет dtype

### [`load_split()`](lib/data.py:450)

**Назначение:** Загрузить индексы сплита из файловой системы.

**Структура директории splits:**
```
splits/
    default/
        train.npy
        val.npy
        test.npy
    a/
        test.npy  # общий test
        0/
            train.npy
            val.npy
        1/
            train.npy
            val.npy
```

### [`apply_split()`](lib/data.py:473)

**Назначение:** Разделить данные по частям сплита.

---

## Преобразования

### [`transform_num()`](lib/data.py:541)

**Политики:**
- `STANDARD` — StandardScaler
- `NOISY_QUANTILE` — QuantileTransformer с шумом

**После трансформации:**
- NaN → 0
- Константные столбцы удаляются

### [`_transform_num_columnwise()`](lib/data.py:597)

**Назначение:** Memory-efficient версия — обрабатывает по одному столбцу.

### [`_extract_bin_from_num()`](lib/data.py:676)

**Назначение:** Извлечь бинарные признаки из числовых.

**Критерий:** Столбец имеет ровно 2 уникальных значения и нет NaN.

### [`transform_cat()`](lib/data.py:752)

**Политики:**
- `ORDINAL` — OrdinalEncoder
- `ONE_HOT` — OrdinalEncoder → OneHotEncoder

**Обработка неизвестных:** В val/test неизвестные категории получают индекс `max_train + 1`.

---

## Task и метрики

### [`Task`](lib/data.py:847)

**Поля:**
- `labels` — метки для каждой части
- `type_` — TaskType (REGRESSION, BINCLASS, MULTICLASS)
- `score` — Score (ACCURACY, ROC_AUC, RMSE и др.)

**Методы:**
- [`is_regression`](lib/data.py:874) — проверка типа задачи
- [`compute_n_classes()`](lib/data.py:889) — количество классов
- [`calculate_metrics()`](lib/data.py:896) — вычисление метрик

### [`Score`](lib/data.py:828)

Перечисление доступных метрик:
- `ACCURACY`
- `CROSS_ENTROPY`
- `MAE`
- `R2`
- `RMSE`
- `ROC_AUC`

---

## Dataset

### [`Dataset`](lib/data.py:920)

**Поля:**
- `data` — словарь `{data_key: {part_key: array}}`
- `task` — объект Task

**Методы:**
- [`from_dir()`](lib/data.py:931) — загрузка из директории
- [`to_torch()`](lib/data.py:937) — конвертация в torch tensors
- [`size()`](lib/data.py:965) — размер части
- [`compute_cat_cardinalities()`](lib/data.py:975) — кардинальности категорий
- [`try_standardize_labels_()`](lib/data.py:1016) — стандартизация для регрессии

---

## [`build_dataset()`](lib/data.py:1023)

**Назначение:** Фабричная функция для создания датасета.

**Параметры:**
- `path` — путь к данным
- `split_id` — ID сплита
- `extract_bin_from_num` — извлекать бинарные
- `skip_bin_encoder` — пропускать encoder для бинарных
- `num_policy` — политика для числовых
- `num_memory_efficient` — memory-efficient режим
- `bin_policy` — политика для бинарных
- `cat_policy` — политика для категориальных
- `task_score` — метрика задачи
- `seed` — случайное зерно
- `cache` — кэширование результата

**Порядок операций:**
1. Загрузка данных
2. Извлечение бинарных признаков
3. Трансформация числовых
4. Трансформация бинарных
5. Трансформация категориальных

---

## Схема данных

```
Директория датасета:
├── info.json       # Метаданные задачи
├── x_num.npy       # Числовые признаки
├── x_bin.npy       # Бинарные признаки (опционально)
├── x_cat.npy       # Категориальные признаки (опционально)
├── y.npy           # Цели
└── splits/         # Сплиты
    └── default/
        ├── train.npy
        ├── val.npy
        └── test.npy
```

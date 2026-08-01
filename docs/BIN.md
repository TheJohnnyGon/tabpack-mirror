# Папка bin/ — Бинарные скрипты

## Назначение

Содержит исполняемые скрипты и бизнес-логику проекта.

## Структура

```
bin/
├── tune.py                  # ⚠️ УСТАРЕВШИЙ — используйте HyperparameterSampler в tabpack.py
└── tabpack/                 # Основной модуль TabPack
    ├── __init__.py
    ├── tabpack.py           # ✅ PRODUCTION — Главное обучение (см. TABPACK_MAIN.md)
    ├── nn.py                # ✅ PRODUCTION — Neural Network модули (см. NN_MODULE.md)
    ├── optim.py             # ✅ PRODUCTION — Оптимизаторы (см. OPTIM_ENSEMBLE.md)
    ├── metrics.py           # ✅ PRODUCTION — Метрики для pack (NumPy)
    ├── metrics_torch.py     # ✅ PRODUCTION — Метрики для pack (Torch)
    ├── ensemble_utils.py    # ✅ PRODUCTION — Ансамблирование (NumPy)
    ├── ensemble_utils_torch.py  # ✅ PRODUCTION — Ансамблирование (Torch)
    ├── ensemble_checkpoint.py   # ✅ PRODUCTION — Хранение чекпоинтов
    ├── types.py             # ✅ PRODUCTION — ConfigDict тип
    ├── utils.py             # ✅ PRODUCTION — Вспомогательные функции
    ├── predict.py           # ⚠️ УСТАРЕВШИЙ — старая реализация инференса, используйте infer/infer.py
    ├── run.py               # ⚠️ УСТАРЕВШИЙ — pipeline для экспериментов на сидах, не используется
    ├── make_configs.py      # ⚠️ УСТАРЕВШИЙ — для статьи, не используется в production
    ├── make_configs_new.py  # ⚠️ УСТАРЕВШИЙ — для статьи, не используется в production
    ├── make_configs_branch.py   # ⚠️ УСТАРЕВШИЙ — для статьи, не используется в production
    └── make_evaluation_config.py # ⚠️ УСТАРЕВШИЙ — для статьи, не используется в production
```

## Production скрипты

**Используются в production:**
- [`bin/tabpack/tabpack.py`](../bin/tabpack/tabpack.py:1) — главное обучение
- [`infer/infer.py`](../infer/infer.py:1) — инференс из model.pt

**Устаревшие (не используются):**
- `bin/tune.py` — используйте `HyperparameterSampler` в `tabpack.py`
- `bin/tabpack/predict.py` — используйте `infer/infer.py`
- `bin/tabpack/run.py` — pipeline для экспериментов на сидах
- `bin/tabpack/make_configs*.py` — для статьи, не нужны в production

---

## [`bin/tune.py`](bin/tune.py:1) — Hyperparameter Tuning ⚠️ УСТАРЕВШИЙ

**Статус:** Устаревший. Используйте [`HyperparameterSampler`](../bin/tabpack/tabpack.py:174) в `tabpack.py` вместо этого.

### Назначение

Обёртка над Optuna для поиска гиперпараметров с поддержкой параллельных воркеров.

### [`Config`](bin/tune.py:341)

| Параметр | Описание |
|----------|----------|
| `seed` | Случайное зерно |
| `function` | Полное имя функции (например 'bin.tabpack.tabpack.main') |
| `space` | Пространство гиперпараметров |
| `n_trials` | Количество trial |
| `timeout` | Таймаут |
| `sampler` | Конфиг сэмплера (TPESampler и др.) |
| `n_workers` | Количество параллельных процессов |
| `n_gpus_per_worker` | GPU на воркера |

### Space формат

```python
space = {
    'model': {
        'n_blocks': ['_tune_', 'int', 1, 5],
        'd_block': ['_tune_', 'loguniform', 32, 512],
        'dropout': ['_tune_', '?uniform', 0.0, 0.0, 1.0],  # ? = опционально
    }
}
```

**Дистрибуции:**
- `'int'` — suggest_int(min, max[, step])
- `'uniform'` — suggest_float(min, max)
- `'loguniform'` — suggest_float(min, max, log=True)
- `'categorical'` — suggest_categorical(choices)
- `'?'` префикс — опциональный параметр

### Ключевые функции

#### [`_sample_config()`](bin/tune.py:88)
Рекурсивно сэмплирует конфиг из space.

#### [`_objective()`](bin/tune.py:164)
Objective функция для Optuna:
1. Сэмплирует конфиг
2. Запускает эксперимент
3. Возвращает val score

#### [`_callback()`](bin/tune.py:259)
Callback после каждого trial:
1. Сохраняет checkpoint
2. Обновляет report
3. Backup

#### [`_worker_main()`](bin/tune.py:412)
Главная функция воркера для параллельного tuning.

---

## [`bin/tabpack/types.py`](bin/tabpack/types.py:1)

```python
type ConfigDict = dict[str, Any]
```

---

## [`bin/tabpack/utils.py`](bin/tabpack/utils.py:1) — Утилиты

### Файлы и время

| Функция | Описание |
|---------|----------|
| [`get_path_from_config()`](bin/tabpack/utils.py:10) | Относительный/абсолютный путь |
| [`time_now()`](bin/tabpack/utils.py:18) | time.perf_counter() |
| [`time_elapsed_since()`](bin/tabpack/utils.py:31) | Разница во времени |

### Dict утилиты

| Функция | Описание |
|---------|----------|
| [`unflatten_dict()`](bin/tabpack/utils.py:40) | Обратное к flatten |
| [`dict_merge_recursively()`](bin/tabpack/utils.py:51) | Рекурсивное слияние |
| [`transpose_list_of_dicts()`](bin/tabpack/utils.py:66) | `[{a:1},{a:2}]` → `{a:[1,2]}` |

### NumPy утилиты

| Функция | Описание |
|---------|----------|
| [`to_numpy()`](bin/tabpack/utils.py:76) | Конвертация в numpy |
| [`numpy_map()`](bin/tabpack/utils.py:102) | Применить fn ко всем array |
| [`numpy_index()`](bin/tabpack/utils.py:113) | Индексирование |
| [`numpy_stack()`](bin/tabpack/utils.py:117) | Stack по всем array |
| [`numpy_concatenate()`](bin/tabpack/utils.py:132) | Concatenate по всем array |

---

## [`bin/tabpack/metrics.py`](bin/tabpack/metrics.py:1) — Метрики (NumPy Pack)

### [`calculate_metrics_pack()`](bin/tabpack/metrics.py:12)

**Вход:**
- `y_true`: `(samples,)`
- `y_pred`: `(pack_size, samples[, classes])`

**Выход:** `Metrics = dict[str, np.ndarray]` где каждый array имеет форму `(pack_size,)`

**Метрики:**
- Регрессия: rmse, r2
- Binclass: accuracy, roc-auc / cross-entropy
- Multiclass: accuracy, cross-entropy

**Особенность:** y_true replicates до pack_size для векторизации.

---

## [`bin/tabpack/metrics_torch.py`](bin/tabpack/metrics_torch.py:1) — Метрики (Torch Pack)

### [`roc_auc_score()`](bin/tabpack/metrics_torch.py:14)
Torch-реализация ROC-AUC (точность до 1e-6 от sklearn).

### [`multiclass_cross_entropy()`](bin/tabpack/metrics_torch.py:66)
Cross-entropy для multiclass без reduction.

### [`calculate_metrics_pack()`](bin/tabpack/metrics_torch.py:84)
Torch-версия метрик для pack.

---

## [`bin/tabpack/ensemble_utils.py`](bin/tabpack/ensemble_utils.py:1) — Ансамблирование (NumPy)

NumPy-версия алгоритмов ансамблирования.

### Алгоритмы

| Функция | Описание |
|---------|----------|
| [`topk_ensemble()`](bin/tabpack/ensemble_utils.py:67) | Top-K по score |
| [`bruteforce_ensemble()`](bin/tabpack/ensemble_utils.py:76) | Полный перебор |
| [`greedy_ensemble()`](bin/tabpack/ensemble_utils.py:102) | Жадный с весами |
| [`greedy_remove_ensemble()`](bin/tabpack/ensemble_utils.py:198) | Жадное удаление |

### Online Ensembles

| Класс | Описание |
|-------|----------|
| [`SimpleOnlineEnsemble`](bin/tabpack/ensemble_utils.py:288) | Простое среднее |
| [`TopKOnlineEnsemble`](bin/tabpack/ensemble_utils.py:328) | Top-K онлайн |
| [`BruteForceOnlineEnsemble`](bin/tabpack/ensemble_utils.py:333) | Bruteforce онлайн |
| [`GreedyOnlineEnsemble`](bin/tabpack/ensemble_utils.py:338) | Greedy онлайн |
| [`GreedyRemoveOnlineEnsemble`](bin/tabpack/ensemble_utils.py:349) | Greedy remove онлайн |

---

## [`bin/tabpack/ensemble_utils_torch.py`](bin/tabpack/ensemble_utils_torch.py:1) — Ансамблирование (Torch)

Torch-версия с дополнительными алгоритмами.

### Алгоритмы

| Функция | Описание |
|---------|----------|
| [`topk_ensemble()`](bin/tabpack/ensemble_utils_torch.py:97) | Top-K |
| [`autotopk_ensemble()`](bin/tabpack/ensemble_utils_torch.py:110) | Автоопределение размера |
| [`bruteforce_ensemble()`](bin/tabpack/ensemble_utils_torch.py:141) | Полный перебор |
| [`greedy_ensemble()`](bin/tabpack/ensemble_utils_torch.py:160) | Жадный с весами и double scoring |
| [`beam_ensemble()`](bin/tabpack/ensemble_utils_torch.py:339) | Beam search |

---

## [`bin/tabpack/ensemble_checkpoint.py`](bin/tabpack/ensemble_checkpoint.py:1) — Чекпоинты ансамбля

### [`EnsembleCheckpointStore`](bin/tabpack/ensemble_checkpoint.py:19)

**Назначение:** Хранение чекпоинтов по ключу `(model_id, step)`.

**Методы:**
- [`save_checkpoint()`](bin/tabpack/ensemble_checkpoint.py:40) — Сохранить
- [`get_checkpoint()`](bin/tabpack/ensemble_checkpoint.py:64) — Получить
- [`has_checkpoint()`](bin/tabpack/ensemble_checkpoint.py:82) — Проверить
- [`remove_checkpoint()`](bin/tabpack/ensemble_checkpoint.py:125) — Удалить
- [`get_all_checkpoints()`](bin/tabpack/ensemble_checkpoint.py:94) — Все чекпоинты
- [`get_unique_model_ids()`](bin/tabpack/ensemble_checkpoint.py:102) — Уникальные ID

---

## Зависимости bin/tabpack/

```
tabpack.py
├── nn.py
├── optim.py → nn.py
├── metrics.py → nn.py (BATCH_DIM, PACK_DIM)
├── metrics_torch.py → nn.py
├── ensemble_utils.py → metrics.py, utils.py
├── ensemble_utils_torch.py → metrics_torch.py, utils.py
├── ensemble_checkpoint.py
├── types.py
└── utils.py

Все модули tabpack/ импортируют:
├── lib.data
├── lib.types
├── lib.util
├── lib.env
├── lib.experiment
├── lib.deep
└── delu (внешняя библиотека)
```

# Документация TabPack

Полная документация проекта TabPack — системы параллельного обучения ансамблей нейронных сетей для табличных данных.

## Суть проекта (аналогия с CatBoost)

**TabPack — это единая модель-ансамбль**, а не коллекция независимых моделей.

| | CatBoost | TabPack |
|---|---|---|
| Базовый элемент | Дерево решений | MLP-сеть |
| Модель | Ансамбль из множества деревьев | Ансамбль из множества MLP |
| Обучение | Последовательное (gradient boosting) | Параллельное (pack-тензоры) |
| Результат | Усреднение предсказаний деревьев | Усреднение предсказаний MLP |

Каждая MLP в ансамбле имеет **свои гиперпараметры** (глубина, ширина, dropout, learning rate), но все они обучаются на одних данных и объединяются в финальном предсказании. TabPack — это одна модель, где "веса" — это веса всех MLP вместе, а "инференс" — это прогон через весь ансамбль и усреднение.

---

## Навигация

### 0. Общий обзор

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Общая архитектура проекта, ключевые классы, зависимости, поток данных

### 1. Критически важные документы (для внесения изменений)

| Файл | Описание |
|------|----------|
| **[CONFIG.md](CONFIG.md)** | 🔴 **Полная документация конфига** — все поля Config TypedDict, матрица конфликтов, подводные камни |
| **[DATA_FLOW.md](DATA_FLOW.md)** | 🔴 **Поток данных** — configs_T, prediction_type, regression_label_stats через все модули |
| **[GOTCHAS.md](GOTCHAS.md)** | 🔴 **Подводные камни** — 25 критических подводных камней, неявные зависимости, что НЕЛЬЗЯ делать |

### 2. Папка lib/ — Базовая библиотека

- **[LIB.md](LIB.md)** — Обобщающая документация по всей папке lib/
  - [`lib/types.py`](lib/types.py:1) — TaskType, PredictionType, KWArgs
  - [`lib/env.py`](lib/env.py:1) — get_project_dir(), is_local()
  - [`lib/util.py`](lib/util.py:1) — init(), get_device(), dataclass_from_dict(), adjust_gpu_memory_usage()
  - [`lib/experiment.py`](lib/experiment.py:1) — run(), finish(), dump_report(), checkpoint
  - [`lib/deep.py`](lib/deep.py:1) — make_parameter_groups(), default_zero_weight_decay_condition()
  - [`lib/metrics.py`](lib/metrics.py:1) — calculate_metrics() (NumPy)
  - [`lib/parallel.py`](lib/parallel.py:1) — map(), get_worker_id(), lock()

### 3. Папка bin/ — Бинарные скрипты

- **[BIN.md](BIN.md)** — Обобщающая документация по всей папке bin/
  - [`bin/tune.py`](bin/tune.py:1) — Hyperparameter tuning (Optuna)
  - [`bin/tabpack/types.py`](bin/tabpack/types.py:1) — ConfigDict
  - [`bin/tabpack/utils.py`](bin/tabpack/utils.py:1) — numpy_stack(), dict_merge_recursively()
  - [`bin/tabpack/metrics.py`](bin/tabpack/metrics.py:1) — calculate_metrics_pack() NumPy
  - [`bin/tabpack/metrics_torch.py`](bin/tabpack/metrics_torch.py:1) — calculate_metrics_pack() Torch
  - [`bin/tabpack/ensemble_utils.py`](bin/tabpack/ensemble_utils.py:1) — Ансамблирование NumPy
  - [`bin/tabpack/ensemble_utils_torch.py`](bin/tabpack/ensemble_utils_torch.py:1) — Ансамблирование Torch
  - [`bin/tabpack/ensemble_checkpoint.py`](bin/tabpack/ensemble_checkpoint.py:1) — EnsembleCheckpointStore

### 4. Детальная документация по модулям

| Файл | Описание |
|------|----------|
| **[TABPACK_MAIN.md](TABPACK_MAIN.md)** | Главный файл обучения — ModelPack, StatePack, OnlineEnsemble, main() training loop |
| **[NN_MODULE.md](NN_MODULE.md)** | Neural Network модули — ModulePack, LinearPack, MLPBackbonePack, embedding модули |
| **[DATA_MODULE.md](DATA_MODULE.md)** | lib/data.py — DataPreprocessor, Dataset, загрузка и преобразование |
| **[INFER_MODULE.md](INFER_MODULE.md)** | infer/infer.py — загрузка model.pt, восстановление модели, оценка ансамбля |
| **[OPTIM_ENSEMBLE.md](OPTIM_ENSEMBLE.md)** | bin/tabpack/optim.py + ensemble_utils — AdamWPack, MuonAdamWPack, алгоритмы ансамблей |

---

## Быстрый старт

1. **[ARCHITECTURE.md](../ARCHITECTURE.md)** — общее понимание
2. **[CONFIG.md](CONFIG.md)** — конфигурация (обязательно для внесения изменений!)
3. **[DATA_FLOW.md](DATA_FLOW.md)** — поток данных через систему
4. **[GOTCHAS.md](GOTCHAS.md)** — подводные камни (обязательно перед изменениями!)
5. **[LIB.md](LIB.md)** — базовая библиотека
6. **[BIN.md](BIN.md)** — бинарные скрипты
7. **[TABPACK_MAIN.md](TABPACK_MAIN.md)** — цикл обучения
8. **[NN_MODULE.md](NN_MODULE.md)** — архитектура моделей

---

## Для внесения изменений (для агентов)

### Критически важные документы (читать ПЕРЕД изменениями):

| Документ | Зачем читать |
|----------|--------------|
| **[CONFIG.md](CONFIG.md)** | Понять все поля конфига, их взаимодействия и конфликты |
| **[DATA_FLOW.md](DATA_FLOW.md)** | Понять как данные проходят через систему (configs_T, prediction_type, regression_label_stats) |
| **[GOTCHAS.md](GOTCHAS.md)** | Избежать 25 критических подводных камней |

### Что меняю → Что смотрю:

| Что меняю | Смотрю |
|-----------|--------|
| Модель (архитектура) | NN_MODULE.md, TABPACK_MAIN.md (ModelPack) |
| Обучение (training loop) | TABPACK_MAIN.md (main()), CONFIG.md (n_epochs, patience) |
| Данные (препроцессинг) | DATA_MODULE.md (DataPreprocessor), DATA_FLOW.md (regression_label_stats) |
| Инференс | INFER_MODULE.md, DATA_FLOW.md (model.pt format) |
| Оптимизатор | OPTIM_ENSEMBLE.md (AdamWPack, MuonAdamWPack), CONFIG.md (optimizer) |
| Ансамблирование | OPTIM_ENSEMBLE.md (алгоритмы), CONFIG.md (online_ensembles), GOTCHAS.md (ensemble snapshot) |
| Метрики | BIN.md (metrics.py, metrics_torch.py), GOTCHAS.md (prediction_type) |
| Hyperparameter tuning | BIN.md (bin/tune.py), CONFIG.md (sampler, configs_T) |
| Эксперименты | LIB.md (lib/experiment.py) |
| Утилиты | LIB.md (lib/util.py), BIN.md (utils.py) |
| Типы данных | LIB.md (lib/types.py) |
| Параллельное выполнение | LIB.md (lib/parallel.py) |

### Чеклист перед внесением изменений:

1. ✅ Прочитать **CONFIG.md** — понять все поля конфига
2. ✅ Прочитать **DATA_FLOW.md** — понять поток данных
3. ✅ Прочитать **GOTCHAS.md** — избежать подводных камней
4. ✅ Прочитать соответствующий модульный документ (TABPACK_MAIN.md, NN_MODULE.md и т.д.)
5. ✅ Проверить матрицу конфликтов в CONFIG.md
6. ✅ Проверить чеклист в GOTCHAS.md для вашего типа изменений

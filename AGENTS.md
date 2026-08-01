# AGENTS.md — Руководство для агентов

Этот документ является **entry point** для всех агентов, работающих с кодом TabPack.

---

## 🚨 КРИТИЧЕСКИ ВАЖНО: Чеклист перед внесением изменений

**ПЕРЕД ЛЮБЫМИ ИЗМЕНЕНИЯМИ** выполните следующие шаги:

### 1. Прочитать критические документы (ОБЯЗАТЕЛЬНО)

- [ ] **[CONFIG.md](docs/CONFIG.md)** — Все поля конфига, матрица конфликтов, подводные камни
- [ ] **[GOTCHAS.md](docs/GOTCHAS.md)** — 25 критических подводных камней, что НЕЛЬЗЯ делать
- [ ] **[DATA_FLOW.md](docs/DATA_FLOW.md)** — Поток данных через систему (configs_T, prediction_type, regression_label_stats)

### 2. Найти функцию в карте зависимостей

- [ ] **[FUNCTION_MAP.md](docs/FUNCTION_MAP.md)** — Полная карта всех функций и их зависимостей

### 3. Прочитать модульную документацию

В зависимости от того, что меняете:

| Что меняю | Читаю |
|-----------|-------|
| Модель (архитектура) | [NN_MODULE.md](docs/NN_MODULE.md), [TABPACK_MAIN.md](docs/TABPACK_MAIN.md) |
| Обучение (training loop) | [TABPACK_MAIN.md](docs/TABPACK_MAIN.md), [CONFIG.md](docs/CONFIG.md) |
| Данные (препроцессинг) | [DATA_MODULE.md](docs/DATA_MODULE.md), [DATA_FLOW.md](docs/DATA_FLOW.md) |
| Инференс | [INFER_MODULE.md](docs/INFER_MODULE.md), [DATA_FLOW.md](docs/DATA_FLOW.md) |
| Оптимизатор | [OPTIM_ENSEMBLE.md](docs/OPTIM_ENSEMBLE.md), [CONFIG.md](docs/CONFIG.md) |
| Ансамблирование | [OPTIM_ENSEMBLE.md](docs/OPTIM_ENSEMBLE.md), [GOTCHAS.md](docs/GOTCHAS.md) |
| Метрики | [BIN.md](docs/BIN.md), [GOTCHAS.md](docs/GOTCHAS.md) |

### 4. Проверить матрицу конфликтов

- [ ] Проверить матрицу конфликтов в [CONFIG.md](docs/CONFIG.md#матрица-конфликтов-и-зависимостей)
- [ ] Проверить чеклист в [GOTCHAS.md](docs/GOTCHAS.md#чеклист-перед-внесением-изменений)

---

## 🗺️ Карта зависимостей

### Главные файлы

| Файл | Назначение | Документация |
|------|------------|--------------|
| [`bin/tabpack/tabpack.py`](bin/tabpack/tabpack.py:1) | Главный файл обучения | [TABPACK_MAIN.md](docs/TABPACK_MAIN.md) |
| [`infer/infer.py`](infer/infer.py:1) | Инференс моделей | [INFER_MODULE.md](docs/INFER_MODULE.md) |
| [`lib/data.py`](lib/data.py:1) | Работа с данными | [DATA_MODULE.md](docs/DATA_MODULE.md) |

### Вспомогательные файлы

| Файл | Назначение | Документация |
|------|------------|--------------|
| [`bin/tabpack/nn.py`](bin/tabpack/nn.py:1) | Neural Network модули | [NN_MODULE.md](docs/NN_MODULE.md) |
| [`bin/tabpack/optim.py`](bin/tabpack/optim.py:1) | Оптимизаторы | [OPTIM_ENSEMBLE.md](docs/OPTIM_ENSEMBLE.md) |
| [`bin/tabpack/ensemble_utils_torch.py`](bin/tabpack/ensemble_utils_torch.py:1) | Алгоритмы ансамблирования | [OPTIM_ENSEMBLE.md](docs/OPTIM_ENSEMBLE.md) |
| [`bin/tabpack/metrics.py`](bin/tabpack/metrics.py:1) | Метрики (NumPy) | [BIN.md](docs/BIN.md) |
| [`bin/tabpack/metrics_torch.py`](bin/tabpack/metrics_torch.py:1) | Метрики (Torch) | [BIN.md](docs/BIN.md) |

### Полная карта

См. **[FUNCTION_MAP.md](docs/FUNCTION_MAP.md)** — полная карта всех функций и зависимостей.

---

## 📋 Типичные задачи

### Задача 1: Добавить новый loss

**Шаги:**

1. **Прочитать:**
   - [GOTCHAS.md#1](docs/GOTCHAS.md#1-loss-scaling-sum-vs-mean) — Loss scaling
   - [FUNCTION_MAP.md#_make_loss_fn_pack](docs/FUNCTION_MAP.md#_make_loss_fn_pack-строка-1046)

2. **Изменить:**
   - [`_make_loss_fn_pack()`](bin/tabpack/tabpack.py:1046) — добавить новый loss

3. **Проверить:**
   - Loss возвращает тензор формы `(pack_size,)` после `mean(BATCH_DIM)`
   - `loss.sum()` в training loop даёт правильные градиенты
   - Метрики в [`metrics.py`](bin/tabpack/metrics.py:1) и [`metrics_torch.py`](bin/tabpack/metrics_torch.py:1) обновлены
   - `prediction_type` совместим с новым loss

**Подводные камни:**
- НЕ использовать `.mean()` вместо `.sum()` — градиенты уменьшатся в pack_size раз
- См. [GOTCHAS.md#1](docs/GOTCHAS.md#1-loss-scaling-sum-vs-mean)

---

### Задача 2: Добавить новый optimizer

**Шаги:**

1. **Прочитать:**
   - [OPTIM_ENSEMBLE.md](docs/OPTIM_ENSEMBLE.md) — Оптимизаторы
   - [FUNCTION_MAP.md#_make_optimizer](docs/FUNCTION_MAP.md#_make_optimizer-строка-1034)

2. **Изменить:**
   - Добавить класс в [`optim.py`](bin/tabpack/optim.py:1)
   - Обновить [`_make_optimizer()`](bin/tabpack/tabpack.py:1034)

3. **Проверить:**
   - Optimizer поддерживает `pack_size` параметр
   - Per-model гиперпараметры передаются как списки
   - `optimizer_pack_remove()` работает корректно
   - `configs_T` flow обновлён если нужны per-model параметры

**Подводные камни:**
- Стандартные оптимизаторы не поддерживают `pack_size` — он автоматически удаляется
- См. [GOTCHAS.md#23](docs/GOTCHAS.md#23-_make_optimizer-pack_size-параметр)

---

### Задача 3: Добавить новый ensemble алгоритм

**Шаги:**

1. **Прочитать:**
   - [OPTIM_ENSEMBLE.md](docs/OPTIM_ENSEMBLE.md) — Алгоритмы ансамблирования
   - [FUNCTION_MAP.md#get_ensemble_fn](docs/FUNCTION_MAP.md#get_ensemble_fn-строка-434)

2. **Изменить:**
   - Добавить функцию в [`ensemble_utils_torch.py`](bin/tabpack/ensemble_utils_torch.py:1)
   - Обновить [`get_ensemble_fn()`](bin/tabpack/ensemble_utils_torch.py:434)

3. **Проверить:**
   - Алгоритм возвращает `(ensemble_idx, ensemble_weights)`
   - `OnlineEnsemble` обновлён если нужны новые параметры
   - Ensemble snapshot сохраняется правильно
   - `update_type` совместим с новым алгоритмом

**Подводные камни:**
- Для `update_type='latest'` ансамбль строится на текущих предсказаниях, но веса продолжают меняться
- См. [GOTCHAS.md#8](docs/GOTCHAS.md#8-save_model-и-ensemble-snapshot)

---

### Задача 4: Изменить модель

**Шаги:**

1. **Прочитать:**
   - [NN_MODULE.md](docs/NN_MODULE.md) — Neural Network модули
   - [FUNCTION_MAP.md#ModelPack](docs/FUNCTION_MAP.md#modelpack-строка-58)

2. **Изменить:**
   - Обновить [`ModelPack`](bin/tabpack/tabpack.py:58)

3. **Проверить:**
   - `forward()` возвращает правильную форму `(pack_size, batch_size, output_dim)`
   - `PACK_DIM=0`, `BATCH_DIM=1` соблюдаются
   - `module_pack_load_state_dict()` работает с новой структурой
   - `_prepare_model_config()` обрабатывает новые per-model параметры
   - `max_*` размерности выводятся правильно

**Подводные камни:**
- Все тензоры имеют форму `(pack_size, batch_size, features)` — НЕ перепутать pack и batch
- См. [GOTCHAS.md#2](docs/GOTCHAS.md#2-pack_dim--0-batch_dim--1)

---

### Задача 5: Изменить данные

**Шаги:**

1. **Прочитать:**
   - [DATA_MODULE.md](docs/DATA_MODULE.md) — Работа с данными
   - [DATA_FLOW.md](docs/DATA_FLOW.md) — Поток данных

2. **Изменить:**
   - Обновить [`DataPreprocessor`](lib/data.py:41)

3. **Проверить:**
   - `fit()` обучает только на train (не на val/test!)
   - `transform()` применяется ко всем частям
   - `regression_label_stats` сохраняется и используется
   - `prediction_type` совместим с новыми данными

**Подводные камни:**
- Fit должен вызываться только на train данных, чтобы избежать data leakage
- См. [GOTCHAS.md#17](docs/GOTCHAS.md#17-datapreprocessor-fit-перед-transform)

---

### Задача 6: Добавить новый флаг в конфиг

**Шаги:**

1. **Прочитать:**
   - [CONFIG.md](docs/CONFIG.md) — Все поля конфига
   - [FUNCTION_MAP.md#Config](docs/FUNCTION_MAP.md#config-строка-1133)

2. **Изменить:**
   - Добавить поле в [`Config`](bin/tabpack/tabpack.py:1133)
   - Обновить [`_validate_config()`](bin/tabpack/tabpack.py:1174) если нужна валидация
   - Добавить использование в [`main()`](bin/tabpack/tabpack.py:1299)

3. **Проверить:**
   - Флаг не конфликтует с другими флагами (см. матрицу конфликтов)
   - Флаг правильно обрабатывается в training loop
   - Флаг сохраняется в `model.pt` если нужен для инференса

**Подводные камни:**
- Некоторые флаги останавливают обучение после первой эпохи (`track_memory_usage`)
- См. [GOTCHAS.md#10](docs/GOTCHAS.md#10-track_memory_usage-останавливает-обучение)

---

## 🔍 Быстрый поиск

### "Я хочу изменить X, какие файлы нужно проверить?"

См. **[FUNCTION_MAP.md](docs/FUNCTION_MAP.md)** — полная карта зависимостей.

### "Что делает этот флаг конфига?"

См. **[CONFIG.md](docs/CONFIG.md)** — все поля конфига с описанием.

### "Какие подводные камни при изменении Y?"

См. **[GOTCHAS.md](docs/GOTCHAS.md)** — 25 критических подводных камней.

### "Как данные проходят через систему?"

См. **[DATA_FLOW.md](docs/DATA_FLOW.md)** — поток данных от конфига до model.pt.

---

## ⚠️ Критические подводные камни (ТОП-5)

### 1. Loss scaling: sum() vs mean()

**Проблема:** В training loop используется `loss.sum()`, а не `loss.mean()`.

**Почему важно:** Если заменить на `.mean()`, градиенты уменьшатся в `pack_size` раз.

**См.:** [GOTCHAS.md#1](docs/GOTCHAS.md#1-loss-scaling-sum-vs-mean)

---

### 2. PACK_DIM = 0, BATCH_DIM = 1

**Проблема:** Все тензоры имеют форму `(pack_size, batch_size, features)`.

**Почему важно:** Если перепутать pack и batch — модели будут использовать чужие веса.

**См.:** [GOTCHAS.md#2](docs/GOTCHAS.md#2-pack_dim--0-batch_dim--1)

---

### 3. Координация pack_remove

**Проблема:** При удалении моделей нужно синхронно обновить **три** компонента: model, optimizer, state.

**Почему важно:** Если не синхронизировать — optimizer будет обновлять несуществующие параметры.

**См.:** [GOTCHAS.md#3](docs/GOTCHAS.md#3-координация-pack_remove)

---

### 4. configs_T должен быть пуст

**Проблема:** После извлечения per-model параметров `configs_T` должен быть пуст.

**Почему важно:** Если остались неиспользованные поля — assert упадёт.

**См.:** [GOTCHAS.md#4](docs/GOTCHAS.md#4-configs_t-должен-быть-пуст)

---

### 5. prediction_type: LABELS vs PROBS vs LOGITS

**Проблема:** Предсказания сохраняются в "aggregation-friendly" формате (LABELS или PROBS, НЕ LOGITS).

**Почему важно:** Ансамбль усредняет предсказания. Усреднение logits не имеет смысла.

**См.:** [GOTCHAS.md#5](docs/GOTCHAS.md#5-prediction_type-labels-vs-probs-vs-logits)

---

## 📚 Навигация по документации

### Критически важные документы (читать ПЕРЕД изменениями)

| Документ | Зачем читать |
|----------|--------------|
| **[CONFIG.md](docs/CONFIG.md)** | Понять все поля конфига, их взаимодействия и конфликты |
| **[DATA_FLOW.md](docs/DATA_FLOW.md)** | Понять как данные проходят через систему |
| **[GOTCHAS.md](docs/GOTCHAS.md)** | Избежать 25 критических подводных камней |

### Модульная документация

| Документ | Описание |
|----------|----------|
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Общая архитектура проекта |
| **[TABPACK_MAIN.md](docs/TABPACK_MAIN.md)** | Главный файл обучения |
| **[NN_MODULE.md](docs/NN_MODULE.md)** | Neural Network модули |
| **[DATA_MODULE.md](docs/DATA_MODULE.md)** | Работа с данными |
| **[INFER_MODULE.md](docs/INFER_MODULE.md)** | Инференс |
| **[OPTIM_ENSEMBLE.md](docs/OPTIM_ENSEMBLE.md)** | Оптимизаторы и ансамбли |
| **[FUNCTION_MAP.md](docs/FUNCTION_MAP.md)** | Полная карта функций и зависимостей |

### Быстрый старт

1. **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — общее понимание
2. **[CONFIG.md](docs/CONFIG.md)** — конфигурация (обязательно для внесения изменений!)
3. **[DATA_FLOW.md](docs/DATA_FLOW.md)** — поток данных через систему
4. **[GOTCHAS.md](docs/GOTCHAS.md)** — подводные камни (обязательно перед изменениями!)
5. **[FUNCTION_MAP.md](docs/FUNCTION_MAP.md)** — карта зависимостей

---

## 🚀 Запуск

### Обучение

```bash
uv run python bin/tabpack/tabpack.py <config.toml> <exp_path>
```

### Инференс

```bash
uv run python infer/infer.py --model-path path/to/model.pt --parts val test
```

### Сравнение с метриками обучения

```bash
uv run python infer/infer.py --model-path path/to/model.pt --compare-nirvana
```

---

## 📖 Дополнительная информация

- **[README.md](docs/README.md)** — Общая документация
- **[BIN.md](docs/BIN.md)** — Бинарные скрипты
- **[LIB.md](docs/LIB.md)** — Базовая библиотека

---

## ❓ FAQ

### Q: Что такое TabPack?

**A:** TabPack — это **единая модель-ансамбль** для табличных данных, обучаемая как пакет параллельных нейронных сетей. Аналогия с CatBoost:
- **CatBoost** = ансамбль из N деревьев → предсказание = усреднение деревьев
- **TabPack** = ансамбль из N MLP → предсказание = усреднение MLP

См. [ARCHITECTURE.md](docs/ARCHITECTURE.md) для деталей.

---

### Q: Что такое pack_size?

**A:** `pack_size` — это количество моделей в ансамбле (параметр `n_models` в конфиге). Все модели обучаются параллельно в одном тензоре формы `(pack_size, batch_size, features)`.

См. [CONFIG.md#n_models](docs/CONFIG.md#n_models-int) для деталей.

---

### Q: Что такое configs_T?

**A:** `configs_T` — это транспонированные конфигурации. Когда используется `sampler` или задан `configs`, каждая модель получает свои гиперпараметры. Для удобства обработки эти конфиги транспонируются:

```python
# Исходные конфиги (список словарей)
state.configs = [
    {'model': {'n_blocks': 1}, 'optimizer': {'lr': 0.001}},
    {'model': {'n_blocks': 2}, 'optimizer': {'lr': 0.002}},
]

# Транспонированные (словарь списков)
configs_T = {
    'model': {'n_blocks': [1, 2]},
    'optimizer': {'lr': [0.001, 0.002]},
}
```

См. [DATA_FLOW.md#1](docs/DATA_FLOW.md#1-поток-configs_t-транспонированные-конфигурации) для деталей.

---

### Q: Почему loss.sum(), а не loss.mean()?

**A:** В training loop используется `loss.sum()`, а не `loss.mean()`, потому что градиенты не должны зависеть от количества моделей в pack. Если использовать `.mean()`, то при увеличении `n_models` градиенты уменьшатся, что сломает обучение.

См. [GOTCHAS.md#1](docs/GOTCHAS.md#1-loss-scaling-sum-vs-mean) для деталей.

---

### Q: Что такое prediction_type?

**A:** `prediction_type` определяет формат предсказаний:
- **LABELS** — денормализованные значения (для регрессии)
- **PROBS** — вероятности после sigmoid/softmax (для классификации)
- **LOGITS** — сырой выход модели (НЕ сохраняется!)

Предсказания сохраняются в "aggregation-friendly" формате, чтобы ансамбль мог их усреднять.

См. [GOTCHAS.md#5](docs/GOTCHAS.md#5-prediction_type-labels-vs-probs-vs-logits) для деталей.

---

## 🎯 Итоговый чеклист

**ПЕРЕД ЛЮБЫМИ ИЗМЕНЕНИЯМИ:**

- [ ] Прочитать [CONFIG.md](docs/CONFIG.md)
- [ ] Прочитать [GOTCHAS.md](docs/GOTCHAS.md)
- [ ] Прочитать [DATA_FLOW.md](docs/DATA_FLOW.md)
- [ ] Найти функцию в [FUNCTION_MAP.md](docs/FUNCTION_MAP.md)
- [ ] Прочитать модульную документацию (см. таблицу выше)
- [ ] Проверить матрицу конфликтов в CONFIG.md
- [ ] Проверить чеклист в GOTCHAS.md

**ПОСЛЕ ИЗМЕНЕНИЙ:**

- [ ] Проверить что все вызываемые функции работают корректно
- [ ] Проверить что `model.pt` формат не изменился (или обновить infer.py)
- [ ] Проверить что метрики вычисляются правильно
- [ ] Проверить что ensemble snapshot сохраняется правильно
- [ ] Запустить обучение и инференс для проверки

---

**Удачи! 🚀**

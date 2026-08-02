# Конфигурация TabPack — Полная документация

## Обзор

Конфигурация TabPack определяется через [`Config`](../bin/tabpack/tabpack.py:1133) TypedDict в [`bin/tabpack/tabpack.py`](../bin/tabpack/tabpack.py:1). Конфиг загружается из `config.toml` и передаётся в функцию [`main()`](../bin/tabpack/tabpack.py:1299).

---

## Обязательные поля

### `seed: int`

**Назначение:** Глобальное случайное зерно для воспроизводимости.

**Использование:**
- [`delu.random.seed(config['seed'])`](../bin/tabpack/tabpack.py:1306) — инициализация генераторов
- [`DataPreprocessor`](../lib/data.py:54) получает seed через `data_config_with_seed`
- [`HyperparameterSampler`](../bin/tabpack/tabpack.py:174) использует seed для Optuna study
- [`batch_generator`](../bin/tabpack/tabpack.py:1489) — `torch.Generator(device).manual_seed(config['seed'])`

**Влияние:** Все случайные операции детерминированы при одинаковом seed.

---

### `data: KWArgs`

**Назначение:** Конфигурация загрузки данных.

**Обязательные ключи:**
- `path: str` — путь к директории с данными (разрешается через `Path().resolve()`)

**Опциональные ключи:**
- `split_id: str` — ID сплита (по умолчанию `lib.data.DEFAULT_SPLIT_ID`)
- Любые параметры [`DataPreprocessor`](../lib/data.py:54): `extract_bin_from_num`, `num_policy`, `cat_policy` и др.

**Обработка:**
```python
data_config_with_seed = {**config['data'], 'seed': config['seed']}
preprocessor = lib.data.DataPreprocessor(data_config_with_seed)
dataset = lib.data.Dataset.from_dir(
    Path(config['data']['path']).resolve(),
    config['data'].get('split_id', lib.data.DEFAULT_SPLIT_ID)
)
dataset = preprocessor.fit_transform(dataset)
```

**Сохранение:** Полный `data_config` с добавленным seed сохраняется в `model.pt` для воспроизводимости инференса.

---

### `n_models: int`

**Назначение:** Общее количество моделей в ансамбле (pack_size).

**Использование:**
- [`StatePack(pack_size=config['n_models'])`](../bin/tabpack/tabpack.py:1368) — размер pack
- Условие продолжения training loop: `report['n_models'] < config['n_models']`
- Валидация: если `configs` задан явно, `len(configs) == n_models`

**Важно:** Это НЕ количество экспериментов, а количество MLP в одном ансамбле. Все модели обучаются параллельно.

---

### `model: KWArgs`

**Назначение:** Конфигурация архитектуры [`ModelPack`](../bin/tabpack/tabpack.py:58).

**Передаётся в:**
```python
ModelPack(
    n_num_features=dataset.n_num_features,
    cat_cardinalities=cat_cardinalities,
    n_classes=n_classes,
    pack_size=state.pack_size,
    **resolved_model_config,
)
```

**Основные ключи:**

| Ключ | Тип | Описание |
|------|-----|----------|
| `num_embeddings` | `dict \| None` | Конфиг embedding числовых признаков |
| `activation` | `str` | Тип активации ('ReLU', 'LeakyReLU' и др.) |
| `d_block` | `int` | Размер скрытого слоя MLP |
| `n_blocks` | `int \| list[int]` | Количество блоков (может быть per-model) |
| `dropout` | `float \| list[float]` | Dropout rate (может быть per-model) |

**num_embeddings варианты:**
- `{'type': 'LinearEmbeddingsPack', 'd_embedding': int}`
- `{'type': 'LinearReLUEmbeddingsPack', 'd_embedding': int, 'concat_input': bool}`
- `{'type': 'CosineEmbeddingsPack', 'd_embedding': int, 'init_scale': float}`

**Автоматический вывод max размерностей:**
- `max_d_embedding` — выводится из `d_embedding` (значение, space или список)
- `max_n_blocks` — выводится из `n_blocks`
- `max_d_block` — выводится из `d_block`

---

### `optimizer: KWArgs`

**Назначение:** Конфигурация оптимизатора.

**Обязательные ключи:**
- `type: str` — тип оптимизатора

**Поддерживаемые типы:**
- `'AdamW'`, `'SGD'` и др. из `torch.optim`
- `'AdamWPack'` — pack-версия AdamW
- `'MuonAdamWPack'` — гибридный Muon + AdamW

**Для AdamWPack/MuonAdamWPack:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `lr` | `float \| list[float]` | Learning rate (может быть per-model) |
| `weight_decay` | `float \| list[float]` | Weight decay (может быть per-model) |
| `beta1`, `beta2` | `float` | Моменты Adam |
| `eps` | `float` | Численная стабильность |
| `shared_step` | `bool` | Общий счётчик шагов для всех моделей |

**Для MuonAdamWPack дополнительно:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `muon_lr` | `float \| list[float]` | Learning rate для Muon |
| `muon_momentum` | `float` | Моментум для Muon |
| `muon_ns_steps` | `int` | Шаги метода Ньютона-Шульца |
| `muon_nesterov` | `bool` | Использовать Nesterov momentum |

**Обработка:**
```python
optimizer = _make_optimizer(
    params=lib.deep.make_parameter_groups(model, ...),
    pack_size=model.pack_size,
    **config['optimizer'],
    **(configs_T.pop('optimizer') if configs_T else {}),
)
```

---

### `batch_size: int`

**Назначение:** Размер тренировочного батча.

**Использование:**
- [`epoch_size = math.ceil(train_size / config['batch_size'])`](../bin/tabpack/tabpack.py:1464)
- [`generate_training_batches(batch_size=config['batch_size'])`](../bin/tabpack/tabpack.py:1576)

**Важно:** Это размер батча для одной модели. Фактический размер тензора: `(pack_size, batch_size, features)`.

---

### `n_epochs: int`

**Назначение:** Максимальное количество эпох для каждой модели.

**Использование:**
```python
stop_pack_idx = compute_stop_pack_idx(
    state,
    epoch_size=epoch_size,
    n_epochs=config['n_epochs'],
    patience=config['patience'],
)
```

**Специальное значение:** `-1` означает бесконечное обучение (остановка только по patience или timeout).

**Логика остановки:**
```python
if n_epochs >= 0 and steps // epoch_size >= n_epochs:
    # Модель достигла лимита эпох
```

---

### `patience: int`

**Назначение:** Количество эпох без улучшения перед остановкой модели.

**Использование:**
- Передаётся в [`compute_stop_pack_idx()`](../bin/tabpack/tabpack.py:519)
- Счётчик `n_consequtive_bad_updates` сравнивается с patience

**Логика:**
```python
if state.n_consequtive_bad_updates[i] > patience:
    # Модель i должна остановиться
```

---

## Опциональные поля

### `eval_parts: NotRequired[list[PartKey]]`

**По умолчанию:** `['val', 'test']`

**Назначение:** Части данных для evaluation после каждой эпохи.

**Ограничение:** Должна содержать `'val'` (assert в коде).

**Использование:**
```python
eval_parts = config.get('eval_parts', ['val', 'test'])
assert 'val' in eval_parts
```

---

### `eval_batch_size: NotRequired[int]`

**По умолчанию:** `32768`

**Назначение:** Размер батча для evaluation (не влияет на training).

**Использование:**
```python
eval_batch_size = config.get('eval_batch_size', 32768)
(eval_metrics, eval_predictions, eval_predictions_torch), eval_batch_size = evaluate(
    parts=eval_parts, batch_size=eval_batch_size
)
```

**Адаптация:** Может автоматически уменьшаться при OOM через `adjust_gpu_memory_usage`.

---

### `online_ensembles: NotRequired[KWArgs]`

**Назначение:** Конфигурация онлайн ансамблей.

**Формат:**
```python
{
    'ensemble_name': {
        'type': str,  # 'greedy', 'autotopk', 'bruteforce', 'beam', 'topk'
        'update_type': 'final' | 'best' | 'latest',
        'patience': int,
        'options': dict,  # Опции алгоритма
        'algorithm_score_fn': None | 'loss',  # Функция оценки
        'include_current_ensemble_in_pool': bool,
    }
}
```

**Пример:**
```python
{
    'greedy': {
        'type': 'greedy',
        'update_type': 'latest',
        'include_current_ensemble_in_pool': True,
        'patience': 32,
        'options': {'max_ensemble_size': 32},
    }
}
```

**Влияние на training loop:**
- Если `online_ensembles is None` — обучение идёт до `n_models` или timeout
- Если задан — обучение продолжается пока хотя бы один ансамбль `is_running`

**update_type:**
- `'final'` — только завершённые модели
- `'best'` — завершённые + лучшие чекпоинты работающих
- `'latest'` — завершённые + текущие предсказания работающих

**include_current_ensemble_in_pool:**

Добавляет модели из **текущего ансамбля** в pool кандидатов для greedy алгоритма.

**Без флага:**
```
Pool = finished_models + running_models (best/latest)
```

**С флагом:**
```
Pool = current_ensemble_models + finished_models + running_models (best/latest)
```

**Пример:**
- Текущий ансамбль: модели `[3, 7, 12]`
- Завершённые модели: `[0, 1, 2, 5, 8]`
- Работающие модели: `[4, 6, 9, 10]`

Без флага: pool = 9 моделей (5 завершённых + 4 работающих)
С флагом: pool = 12 моделей (3 текущих + 5 завершённых + 4 работающих)

**Зачем нужно:**
1. **Переиспользование** — greedy может оставить модели из текущего ансамбля
2. **Стабильность** — ансамбль не меняется радикально каждую эпоху
3. **With replacement** — модель из текущего ансамбля может добавиться несколько раз

**Подводный камень:** При `update_type='latest'` текущий ансамбль использует **текущие предсказания** (не лучшие!), но веса продолжают меняться. Поэтому ensemble snapshot сохраняется при каждом улучшении.

---

### `configs: NotRequired[list[ConfigDict]]`

**Назначение:** Предопределённые конфигурации для каждой модели.

**Взаимоисключение:** Нельзя использовать одновременно с `sampler`.

**Валидация:**
```python
if configs is not None:
    assert len(configs) == config['n_models']
```

**Использование:**
```python
state = StatePack(
    pack_size=config['n_models'],
    configs=_prepare_configs(config, hyperparameter_sampler),
)
```

**Когда использовать:** Для воспроизводимости или когда нужно задать конкретные гиперпараметры для каждой модели.

---

### `sampler: NotRequired[KWArgs]`

**Назначение:** Конфигурация [`HyperparameterSampler`](../bin/tabpack/tabpack.py:174) для автоматической генерации конфигов.

**Взаимоисключение:** Нельзя использовать одновременно с `configs`.

**Формат:**
```python
{
    'type': str,  # 'TPESampler', 'RandomSampler', 'GridSampler', 'QMCSampler', 'BruteForceSampler'
    'space': dict,  # Пространство гиперпараметров
    'strict_n_startup_trials': bool,  # Опционально
    # Другие параметры Optuna sampler
}
```

**Пример space:**
```python
{
    'model': {
        'n_blocks': ['_tune_', 'int', 1, 3],
        'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
    },
    'optimizer': {
        'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
        'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
    },
}
```

**Дистрибуции:**
- `'int'` — `suggest_int(min, max[, step])`
- `'uniform'` — `suggest_float(min, max)`
- `'loguniform'` — `suggest_float(min, max, log=True)`
- `'categorical'` — `suggest_categorical(choices)`
- `'?'` префикс — опциональный параметр (с вероятностью выбирает None)

**Влияние на max размерности:**
- Если параметр в space — `max_*` выводится из верхней границы space
- Если параметр задан явно — `max_*` выводится из `max(value_list)`

---

### `amp_dtype: NotRequired[AMPDType]`

**По умолчанию:** `None` (AMP отключен)

**Назначение:** Automatic Mixed Precision для ускорения обучения.

**Поддерживаемые значения:**
- `'bfloat16'` — рекомендуется для современных GPU (A100, H100)
- `'float16'` — **НЕ ПОДДЕРЖИВАЕТСЯ** (assert в коде)

**Ограничение:**
```python
assert dtype is torch.bfloat16, 'For now, only "bfloat16" is supported as amp_dtype'
```

**Использование:**
```python
amp_dtype = config.get('amp_dtype')
if amp_dtype is None:
    autocast = None
else:
    autocast = _make_autocast(amp_dtype, device)
    apply_model = autocast(apply_model_impl)
```

**Влияние:** Ускоряет forward/backward pass, уменьшает использование VRAM.

---

### `timeout: NotRequired[int]`

**По умолчанию:** `None` (без таймаута)

**Назначение:** Максимальное время обучения в секундах.

**Использование:**
```python
timeout = config.get('timeout')
timer = delu.tools.Timer()

while (
    report['n_models'] < config['n_models']
    and (timeout is None or timer.elapsed() < timeout)
    and ...
):
    # Training loop
```

**Взаимодействие:**
- Если `timeout` задан и истёк — обучение останавливается даже если `n_models` не достигнут
- Если `online_ensembles` задан — обучение может остановиться раньше timeout если все ансамбли завершились

---

### `data_on_cpu: NotRequired[bool]`

**По умолчанию:** `False`

**Назначение:** Хранить данные на CPU и перемещать батчи на GPU во время обучения.

**Когда использовать:**
- Когда датасет не помещается в GPU память
- Когда нужно обучать на больших датасетах с ограниченной VRAM
- Когда overhead от CPU→GPU transfer приемлем

**Использование:**
```python
data_on_cpu = config.get('data_on_cpu', False)
if data_on_cpu:
    dataset = dataset.to_torch('cpu', pin_memory=True)
    # Данные остаются на CPU с pinned memory
    # Батчи перемещаются на GPU в apply_model_impl()
else:
    dataset = dataset.to_torch(device)
    # Все данные на GPU
```

**Как работает:**
1. Данные конвертируются в torch тензоры на CPU с `pin_memory=True`
2. Batch индексы генерируются на CPU
3. В [`apply_model_impl()`](../bin/tabpack/tabpack.py:158) данные индексируются на CPU
4. Индексированные батчи перемещаются на GPU через `.to(device, non_blocking=True)`
5. Модель работает с данными на GPU как обычно

**Преимущества:**
- Данные любого размера могут быть использованы (ограничение только RAM)
- Pinned memory ускоряет CPU→GPU transfer
- Non-blocking transfer позволяет overlap с вычислениями

**Недостатки:**
- Overhead от CPU→GPU transfer на каждом батче
- Может быть медленнее для маленьких датасетов, которые помещаются в GPU

**Пример конфигурации:**
```toml
seed = 0
n_models = 64
batch_size = 1024
n_epochs = -1
patience = 16
data_on_cpu = true  # Данные остаются на CPU

[data]
path = "data/large_dataset"

[model]
activation = "SiLU"
d_block = 384

[optimizer]
type = "MuonAdamWPack"
shared_step = true
```

**Взаимодействие с другими параметрами:**
- `amp_dtype` — работает как обычно, AMP применяется к данным на GPU
- `batch_size` — может быть увеличен, так как на GPU только текущий батч
- `eval_batch_size` — evaluation тоже использует CPU данные

---

### `track_experiments: NotRequired[bool]`

**По умолчанию:** `True` если `sampler is None`, иначе `False`

**Назначение:** Трекать ли индивидуальные эксперименты для каждой модели.

**Использование:**
```python
track_experiments = config.get('track_experiments', hyperparameter_sampler is None)

if track_experiments:
    mean_scores, mean_scores_improved = _get_mean_scores(...)
```

**Влияние:**
- `True` — вычисляет средние метрики по всем завершённым моделям
- `False` — пропускает вычисление (экономит время)

**Когда использовать:**
- `True` — для анализа распределения результатов
- `False` — когда важен только финальный ансамбль

---

### `track_best_experiment: NotRequired[bool]`

**По умолчанию:** `True` если `configs is not None`, иначе `False`

**Назначение:** Трекать ли лучшую индивидуальную модель.

**Использование:**
```python
track_best_experiment = config.get(
    'track_best_experiment',
    state.configs is not None,
)

if track_best_experiment:
    best_scores, best_scores_improved = _get_best_scores(...)
```

**Логика дефолта:**
- Если модели имеют разные гиперпараметры (через sampler или configs) — трекаем лучшую
- Если все модели одинаковые — не трекаем (все равноценны)

---

### `track_online_ensemble_history: NotRequired[bool]`

**По умолчанию:** `False`

**Назначение:** Сохранять ли историю обновлений онлайн ансамблей.

**Использование:**
```python
if config.get('track_online_ensemble_history', False):
    report['online_ensembles'][ensemble_name]['report']['history'] = (
        online_ensemble_history[ensemble_name]
    )
```

**Влияние:**
- `True` — в report.json добавляется полная история каждого обновления ансамбля
- `False` — сохраняется только финальное состояние

**Когда использовать:** Для анализа динамики формирования ансамбля.

---

### `track_memory_usage: NotRequired[bool]`

**По умолчанию:** `False`

**Назначение:** Трекать использование GPU памяти.

**Использование:**
```python
if config.get('track_memory_usage', False):
    torch.cuda.reset_peak_memory_stats(device)
    # ... training ...
    report['memory-usage'] = torch.cuda.memory_stats(device)
    break  # Останавливает обучение после первой эпохи!
```

**Важно:** При `track_memory_usage=True` обучение останавливается после первой эпохи для сбора статистики.

**Когда использовать:** Для профилирования и оптимизации использования VRAM.

---

### `save_final_predictions: NotRequired[bool]`

**По умолчанию:** `True`

**Назначение:** Сохранять ли финальные предсказания в `predictions.npz`.

**Использование:**
```python
if config.get('save_final_predictions', True):
    np.savez(exp / 'predictions.npz', **final_state.predictions)
```

**Содержимое:** Предсказания лучших чекпоинтов всех завершённых моделей для всех частей (train, val, test).

---

### `save_all_predictions: NotRequired[bool]`

**По умолчанию:** `False`

**Назначение:** Сохранять ли предсказания после каждой эпохи.

**Использование:**
```python
pack_epochs_numlog.append(
    deepcopy(
        {
            'step': state.steps,
            'time': np.full((state.pack_size,), timer.elapsed()),
            'size': np.array([state.pack_size]),
            'id': state.ids,
            'metrics': eval_metrics,
            **(
                {'predictions': eval_predictions}
                if config.get('save_all_predictions', False)
                else {}
            ),
        }
    )
)
```

**Влияние:**
- `True` — в `numlog.npz` сохраняются предсказания каждой эпохи (большой файл)
- `False` — сохраняются только метрики

**Когда использовать:** Для детального анализа динамики обучения.

---

### `save_model: NotRequired[bool]`

**По умолчанию:** `False`

**Назначение:** Сохранять ли `model.pt` для standalone инференса.

**Использование:**
```python
save_model = config.get('save_model', False)
checkpoint_store = bin.tabpack.ensemble_checkpoint.EnsembleCheckpointStore()

# В training loop:
if save_model:
    for i in map(int, stop_pack_idx):
        member_id = int(state.ids[i])
        member_step = int(state.best_steps[i])
        checkpoint_store.save_checkpoint(...)

# После training:
if save_model:
    torch.save({...}, exp / 'model.pt')
```

**Влияние:**
- `True` — сохраняет чекпоинты всех моделей и создаёт `model.pt`
- `False` — не сохраняет (экономит место и время)

**Когда использовать:**
- `True` — для последующего инференса через `infer.py`
- `False` — для экспериментов, где важен только report

---

## Матрица конфликтов и зависимостей

### Взаимоисключающие параметры

| Параметр 1 | Параметр 2 | Конфликт |
|------------|------------|----------|
| `configs` | `sampler` | Нельзя задать оба одновременно |
| `amp_dtype='float16'` | Любой | Float16 не поддерживается |

### Зависимости параметров

| Параметр | Зависит от | Условие |
|----------|------------|---------|
| `track_experiments` (default) | `sampler` | `True` если `sampler is None` |
| `track_best_experiment` (default) | `configs` | `True` если `configs is not None` |
| `max_n_blocks` | `n_blocks` | Выводится автоматически |
| `max_d_block` | `d_block` | Выводится автоматически |
| `max_d_embedding` | `d_embedding` | Выводится автоматически |

### Критические комбинации

| Комбинация | Эффект | Рекомендация |
|------------|--------|--------------|
| `save_model=False` + `online_ensembles` с `update_type='best'` | Ансамбль использует best weights, но они не сохраняются | Установить `save_model=True` |
| `save_model=False` + `online_ensembles` с `update_type='latest'` | Ансамбль использует текущие weights, которые теряются после обучения | Установить `save_model=True` |
| `n_epochs=-1` + `patience=0` | Модели останавливаются сразу без улучшения | Увеличить patience |
| `timeout` + `online_ensembles` | Обучение может остановиться по timeout до завершения ансамблей | Увеличить timeout или убрать ансамбли |
| `track_memory_usage=True` | Обучение останавливается после первой эпохи | Использовать только для профилирования |

### Влияние на сохранение model.pt

| Параметр | Влияние |
|----------|---------|
| `save_model=True` | Создаёт `model.pt` |
| `online_ensembles` | Определяет какие модели сохраняются (ensemble_info) |
| `data` | Сохраняется в `data_config` для воспроизводимости |
| `seed` | Добавляется в `data_config` |
| `model` | Сохраняется в `model_config` |
| `configs` или `sampler` | Сохраняются в `configs` (все сгенерированные конфиги) |

---

## Примеры конфигураций

### Минимальная конфигурация (без sampler)

```toml
seed = 0
n_models = 8
batch_size = 256
n_epochs = 100
patience = 16

[data]
path = "data/california"

[model]
activation = "ReLU"
d_block = 128
n_blocks = 2
dropout = 0.1

[optimizer]
type = "AdamW"
lr = 0.001
weight_decay = 0.01
```

### Production конфиг: Plain TabPack (без embeddings)

**Реальный конфиг из production (seed=0):**

```toml
seed = 0
n_models = 64
batch_size = 1024
n_epochs = -1
patience = 16
amp_dtype = "bfloat16"
save_all_predictions = false
save_final_predictions = true
track_online_ensemble_history = true
track_experiments = true
save_model = true

[data]
path = "data/peru"
num_policy = "noisy-quantile"
num_memory_efficient = false
extract_bin_from_num = true
skip_bin_encoder = true
bin_policy = "convert-to-cat"
cache = false

[optimizer]
type = "MuonAdamWPack"
shared_step = true

[online_ensembles.greedy]
type = "greedy"
update_type = "latest"
include_current_ensemble_in_pool = true
patience = 8

[online_ensembles.greedy.options]
max_ensemble_size = 32

[model]
activation = "SiLU"
d_block = 384

[sampler]
type = "RandomSampler"

[sampler.space.model]
n_blocks = ["_tune_", "int", 1, 3]
dropout = ["_tune_", "?uniform", 0.0, 0.0, 0.5]

[sampler.space.optimizer]
lr = ["_tune_", "loguniform", 0.0001, 0.005]
weight_decay = ["_tune_", "loguniform", 0.001, 1.0]
muon_lr = ["_tune_", "loguniform", 0.001, 0.1]
```

**Ключевые особенности:**
- `activation = "SiLU"` — SiLU (Swish) вместо ReLU для лучшей производительности
- `patience = 16` для моделей, `patience = 8` для ансамбля — ансамбль останавливается быстрее
- `extract_bin_from_num = true` + `skip_bin_encoder = true` + `bin_policy = "convert-to-cat"` — полный pipeline: извлечение бинарных из числовых → конвертация в категориальные
- `num_policy = "noisy-quantile"` — QuantileTransformer с шумом для числовых признаков
- `save_all_predictions = false` — экономия места (не сохраняются предсказания каждой эпохи)
- `save_final_predictions = true` — сохраняются финальные предсказания лучших чекпоинтов

---

### Production конфиг: TabPack с CosineEmbeddingsPack

**Реальный конфиг из production (seed=1):**

```toml
seed = 1
n_models = 64
batch_size = 1024
n_epochs = -1
patience = 16
amp_dtype = "bfloat16"
save_all_predictions = false
save_final_predictions = true
track_online_ensemble_history = true
track_experiments = true
save_model = true

[data]
path = "data/peru"
num_policy = "noisy-quantile"
num_memory_efficient = false
extract_bin_from_num = true
skip_bin_encoder = true
bin_policy = "convert-to-cat"
cache = false

[optimizer]
type = "MuonAdamWPack"
shared_step = true

[online_ensembles.greedy]
type = "greedy"
update_type = "latest"
include_current_ensemble_in_pool = true
patience = 8

[online_ensembles.greedy.options]
max_ensemble_size = 32

[model]
activation = "SiLU"
d_block = 384

[model.num_embeddings]
type = "CosineEmbeddingsPack"

[sampler]
type = "RandomSampler"

[sampler.space.model]
n_blocks = ["_tune_", "int", 1, 3]
dropout = ["_tune_", "?uniform", 0.0, 0.0, 0.5]

[sampler.space.model.num_embeddings]
d_embedding = ["_tune_", "int", 8, 32, 4]
init_scale = ["_tune_", "loguniform", 0.01, 10.0]

[sampler.space.optimizer]
lr = ["_tune_", "loguniform", 0.0001, 0.005]
weight_decay = ["_tune_", "loguniform", 0.001, 1.0]
muon_lr = ["_tune_", "loguniform", 0.001, 0.1]
```

**Отличия от plain конфига:**
- Добавлен `[model.num_embeddings]` с `type = "CosineEmbeddingsPack"`
- Добавлены `d_embedding` и `init_scale` в sampler space для настройки embeddings
- `d_embedding = ["_tune_", "int", 8, 32, 4]` — размер embedding от 8 до 32 с шагом 4
- `init_scale = ["_tune_", "loguniform", 0.01, 10.0]` — масштаб инициализации весов

---

### Конфигурация с предопределёнными configs

```toml
seed = 0
n_models = 4
batch_size = 256
n_epochs = 50
patience = 10
save_model = true

[data]
path = "data/california"

[model]
activation = "ReLU"
d_block = 128

[optimizer]
type = "AdamW"

[[configs]]
n_blocks = 1
dropout = 0.0
lr = 0.001
weight_decay = 0.01

[[configs]]
n_blocks = 2
dropout = 0.1
lr = 0.0005
weight_decay = 0.005

[[configs]]
n_blocks = 3
dropout = 0.2
lr = 0.0001
weight_decay = 0.001

[[configs]]
n_blocks = 2
dropout = 0.15
lr = 0.002
weight_decay = 0.02
```

---

## Поддерживаемые активации

**Стандартные (из `torch.nn`):**
- `ReLU` — классическая активация
- `SiLU` (Swish) — `x * sigmoid(x)`, часто даёт лучшую производительность
- `GELU` — Gaussian Error Linear Unit
- `LeakyReLU` — с отрицательным наклоном
- `Tanh`, `Sigmoid` и др.

**Pack-активации (из `bin.tabpack.nn`):**
- `LeakyReLUPack` — pack-версия с per-model negative_slope

**Пример использования:**
```toml
[model]
activation = "SiLU"  # или "ReLU", "GELU", "LeakyReLU" и т.д.
```

**Важно:** Активация применяется после каждого линейного слоя в MLPBackbonePack.

---

## Подводные камни

### 1. Loss scaling

В training loop используется `loss = losses.sum()`, а не `.mean()`:
```python
losses = loss_fn(apply_model(...), Y_train[batch_idx])
loss = losses.sum()  # Сумма по pack dimension!
```

**Почему:** Градиенты не должны зависеть от количества моделей в pack. Если использовать `.mean()`, то при увеличении `n_models` градиенты уменьшатся, что сломает обучение.

**Влияние:** При изменении `n_models` нужно пропорционально менять `lr` если используется стандартный optimizer (не Pack).

### 2. Prediction type

```python
prediction_type = (
    PredictionType.LABELS if dataset.task.is_regression else PredictionType.PROBS
)
```

**Важно:** Предсказания сохраняются в "aggregation-friendly" формате:
- Регрессия — labels (денормализованные)
- Классификация — probs (после sigmoid/softmax)

**Не сохранять logits!** Это сломает ансамблирование.

### 3. Regression label standardization

```python
regression_label_stats = dataset.try_standardize_labels_()
```

**Влияние:**
- Для регрессии labels стандартизуются (mean=0, std=1)
- `regression_label_stats` сохраняется в `model.pt`
- При инференсе предсказания денормализуются обратно

**Подводный камень:** Если забыть сохранить `regression_label_stats`, инференс даст неправильные результаты.

### 4. configs_T flow

```python
configs_T = transpose_list_of_dicts(state.configs)
# configs_T['model']['n_blocks'] = [1, 2, 3, 1, 2, 3]  # для 6 моделей

resolved_model_config = _prepare_model_config(config, configs_T.pop('model'))
# resolved_model_config['n_blocks'] = [1, 2, 3, 1, 2, 3]
# resolved_model_config['max_n_blocks'] = 3

optimizer = _make_optimizer(
    ...,
    **config['optimizer'],
    **(configs_T.pop('optimizer') if 'optimizer' in configs_T else {}),
)
```

**Важно:** `configs_T` должен быть пуст после всех `pop()`. Если остались неиспользованные поля — assert:
```python
assert not configs_T, f'The following fields were not used: {", ".join(configs_T)}'
```

### 5. Online ensemble patience

```python
while (
    report['n_models'] < config['n_models']
    and (timeout is None or timer.elapsed() < timeout)
    and (
        online_ensembles is None
        or any(x.is_running for x in online_ensembles.values())
    )
):
```

**Подводный камень:** Если все ансамбли исчерпали patience (`is_running=False`), обучение останавливается даже если `n_models` не достигнут.

### 6. save_model и ensemble snapshot

```python
if first_online_ensemble_improved and save_model:
    # Сохраняем snapshot ансамбля
    # Для update_type='latest' — используем ТЕКУЩИЕ веса
    # Для update_type='best' — используем ЛУЧШИЕ веса
```

**Важно:** Для `update_type='latest'` ансамбль строится на предсказаниях из конкретного момента, но веса продолжают меняться. Snapshot необходим для воспроизводимости.

---

## Чеклист для внесения изменений

### При добавлении нового loss:

1. Обновить [`_make_loss_fn_pack()`](../bin/tabpack/tabpack.py:1046)
2. Проверить что loss возвращает тензор формы `(pack_size,)` после `mean(BATCH_DIM)`
3. Убедиться что `loss.sum()` в training loop даёт правильные градиенты
4. Обновить метрики в [`bin/tabpack/metrics.py`](../bin/tabpack/metrics.py:1) и [`metrics_torch.py`](../bin/tabpack/metrics_torch.py:1)
5. Проверить `prediction_type` — возможно нужен новый тип

### При добавлении нового optimizer:

1. Добавить класс в [`bin/tabpack/optim.py`](../bin/tabpack/optim.py:1)
2. Обновить [`_make_optimizer()`](../bin/tabpack/tabpack.py:1034)
3. Проверить поддержку `pack_size` параметра
4. Если optimizer имеет per-model гиперпараметры — добавить в `configs_T` flow
5. Проверить `optimizer_pack_remove()` для корректного удаления моделей

### При добавлении нового ensemble алгоритма:

1. Добавить функцию в [`bin/tabpack/ensemble_utils_torch.py`](../bin/tabpack/ensemble_utils_torch.py:1)
2. Обновить [`get_ensemble_fn()`](../bin/tabpack/ensemble_utils_torch.py:434)
3. Проверить что алгоритм возвращает `(ensemble_idx, ensemble_weights)`
4. Обновить [`OnlineEnsemble`](../bin/tabpack/tabpack.py:603) если нужны новые параметры
5. Проверить сохранение snapshot в training loop

### При изменении модели:

1. Обновить [`ModelPack`](../bin/tabpack/tabpack.py:58)
2. Проверить что `forward()` возвращает правильную форму
3. Обновить `_prepare_model_config()` если добавлены новые per-model параметры
4. Проверить `module_pack_load_state_dict()` для корректной загрузки весов
5. Обновить `model.pt` формат если изменилась структура

---

## Ссылки

- [`Config` TypedDict](../bin/tabpack/tabpack.py:1133)
- [`main()` функция](../bin/tabpack/tabpack.py:1299)
- [`_validate_config()`](../bin/tabpack/tabpack.py:1174)
- [`_prepare_configs()`](../bin/tabpack/tabpack.py:1190)
- [`_prepare_model_config()`](../bin/tabpack/tabpack.py:1204)
- [`HyperparameterSampler`](../bin/tabpack/tabpack.py:174)
- [`OnlineEnsemble`](../bin/tabpack/tabpack.py:603)

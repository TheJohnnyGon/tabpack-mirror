# Модуль tabpack.py — Главное обучение

## Назначение

Основной скрипт обучения с реализацией полного цикла: инициализация → training loop → evaluation → ensembling → сохранение.

## Ключевая концепция: TabPack = одна модель-ансамбль

Важно понимать: **TabPack — это одна модель**, а не набор независимых экспериментов. Аналогия с CatBoost:

- **CatBoost** = ансамбль из N деревьев → предсказание = усреднение деревьев
- **TabPack** = ансамбль из N MLP → предсказание = усреднение MLP

Параметр `n_models` в конфиге — это не "количество экспериментов", а **количество MLP в одном ансамбле**. Все они обучаются параллельно (pack-тензоры), но в финале представляют собой единую модель. Файл `model.pt` — это артефакт одной модели, а не коллекции.

---

## Классы моделей

### [`ModelPack`](bin/tabpack/tabpack.py:58)

**Полная структура:**

```
ModelPack
├── num_module: LinearEmbeddingsPack | CosineEmbeddingsPack | None
│   └── Преобразует числовые признаки в embeddings
├── cat_module: OneHotEncoding | None
│   └── Кодит категориальные признаки
├── pack_view: PackView
│   └── Расширяет вход до pack-формата
├── backbone: MLPBackbonePack
│   └── Основная нейросеть (MLP)
└── output: LinearPack
    └── Выходной слой (1 для регрессии/binclass, n_classes для multiclass)
```

**[`forward()`](bin/tabpack/tabpack.py:107):**
1. Обработка x_num через num_module (если есть)
2. Обработка x_cat через cat_module (если есть)
3. Конкатенация → pack_view → backbone → output

---

### [`apply_model_impl()`](bin/tabpack/tabpack.py:158)

**Назначение:** Стандартный вызов модели с извлечением данных из Dataset.

**Возвращает:** `Tensor` формы `(batch_size,)` — logits для регрессии/binclass или `(batch_size, n_classes)` для multiclass.

---

## Hyperparameter Sampler

### [`HyperparameterSampler`](bin/tabpack/tabpack.py:174)

**Назначение:** Обёртка над Optuna для генерации конфигураций.

**Методы:**

#### [`ask()`](bin/tabpack/tabpack.py:218)
Создаёт новый trial и сэмплирует конфигурацию из пространства.

#### [`tell()`](bin/tabpack/tabpack.py:228)
Сообщает результат trial (score на валидации).

**Поддерживаемые сэмплеры:**
- `TPESampler` (по умолчанию)
- `RandomSampler`
- `GridSampler`
- `QMCSampler`
- `BruteForceSampler`

---

## State Management

### [`StatePack`](bin/tabpack/tabpack.py:260) — Состояние обучаемых моделей

**Назначение:** Хранит динамическое состояние всех моделей в pack, которые **ещё обучаются**.

**Важно:** Когда модель останавливается (по patience или n_epochs), она удаляется из StatePack и её результаты переносятся в FinalStatePack.

**Динамические свойства:**

| Свойство | Форма | Описание |
|----------|-------|----------|
| `_ids` | `(pack_size,)` | ID моделей |
| `_steps` | `(pack_size,)` | Текущий шаг каждой модели |
| `_n_consequtive_bad_updates` | `(pack_size,)` | Счётчик плохих обновлений |
| `_best_metrics` | `{part: {metric: array}}` | Лучшие метрики за всё время обучения |
| `_best_steps` | `(pack_size,)` | Шаг, на котором достигнуты лучшие метрики |
| `_best_predictions` | `{part: (pack_size, samples)}` | Лучшие предсказания (numpy) |
| `_best_predictions_torch` | `{part: (pack_size, samples)}` | Лучшие предсказания (torch) |
| `_best_model_state_dicts` | `{param_name: tensor}` | Лучшие веса моделей |

**Пример:** Модель 5 обучается 1000 шагов, лучший результат на шаге 800:
```python
state.steps[5] = 1000  # Текущий шаг
state.best_steps[5] = 800  # Лучший шаг
state.best_metrics['val']['score'][5] = 0.95  # Лучшая метрика
state.best_model_state_dicts = {...}  # Веса на шаге 800
```

#### [`update()`](bin/tabpack/tabpack.py:384)

**Логика:**
1. Сравнивает текущие метрики с лучшими (только по val score!)
2. Для улучшенных моделей:
   - Сбрасывает `n_consequtive_bad_updates`
   - Сохраняет новые best_metrics, best_predictions, best_model_state_dicts
3. Для не улучшенных: инкрементирует `n_consequtive_bad_updates`

#### [`remove()`](bin/tabpack/tabpack.py:343)

Удаляет завершённые модели из всех свойств. Вызывается когда модель останавливается.

---

### [`FinalStatePack`](bin/tabpack/tabpack.py:432) — Результаты завершённых моделей

**Назначение:** Хранит результаты моделей, которые **уже остановились** (по patience или n_epochs).

**Важно:** Хранит предсказания **лучших чекпоинтов**, не финальных шагов!

**Свойства:**

| Свойство | Форма | Описание |
|----------|-------|----------|
| `_ids` | `(n_finished,)` | ID завершённых моделей |
| `_steps` | `(n_finished,)` | Лучшие шаги завершённых моделей |
| `_predictions` | `{part: (n_finished, samples)}` | Предсказания лучших чекпоинтов (numpy) |
| `_predictions_torch` | `{part: (n_finished, samples)}` | Предсказания лучших чекпоинтов (torch) |

**Пример:** Модель 5 остановилась на шаге 1200, но лучший результат был на шаге 800:
```python
final_state.ids = [5, ...]
final_state.steps = [800, ...]  # Лучший шаг, не финальный!
final_state.predictions = {...}  # Предсказания на шаге 800
```

**[`extend()`](bin/tabpack/tabpack.py:458):** Добавляет новые завершённые модели.

---

### Разница между StatePack и FinalStatePack

| Аспект | StatePack | FinalStatePack |
|--------|-----------|----------------|
| **Статус моделей** | Ещё обучаются | Уже остановились |
| **Что хранится** | Best metrics + weights | Final predictions |
| **Когда обновляется** | Каждую эпоху | При остановке модели |
| **Использование** | Early stopping, online ensemble | Финальный отчёт, инференс |
| **Pack size** | Уменьшается при остановке | Увеличивается при остановке |

**Поток данных:**
```
Модель обучается → StatePack (best metrics обновляются)
       ↓
Модель остановилась → FinalStatePack (добавляются predictions)
                    → StatePack.remove() (удаляется из pack)
```

---

## Валидация и удаление

### [`pack_validate()`](bin/tabpack/tabpack.py:486)

Проверяет консистентность:
- pack_size модели == pack_size state
- Все ParameterPack/BufferPack имеют правильный pack_size
- optimizer param groups соответствуют pack_size

### [`pack_remove()`](bin/tabpack/tabpack.py:505)

Удаляет модели из model, optimizer и state согласованно.

### [`compute_stop_pack_idx()`](bin/tabpack/tabpack.py:519)

**Критерии остановки:**
1. `patience` — если `n_consequtive_bad_updates > patience`
2. `n_epochs` — если `steps // epoch_size >= n_epochs`

Возвращает индексы моделей для остановки или None.

---

## Эксперименты

### [`assemble_experiments()`](bin/tabpack/tabpack.py:547)

Создаёт список ExperimentDict для завершённых моделей.

### [`get_best_experiment()`](bin/tabpack/tabpack.py:579)

Находит лучший эксперимент по val score.

---

## Online Ensembles

### [`OnlineEnsemble`](bin/tabpack/tabpack.py:603)

**Параметры:**

| Параметр | Описание |
|----------|----------|
| `type` | 'greedy', 'autotopk', 'bruteforce', 'beam', 'topk' |
| `options` | Дополнительные опции алгоритма |
| `algorithm_score_fn` | Функция оценки внутри алгоритма |
| `update_type` | 'final', 'best', 'latest' |
| `update_part` | Часть данных для обновления ('val') |
| `patience` | Терпение перед остановкой |

**[`update()`](bin/tabpack/tabpack.py:749):**

1. Формирует пул кандидатов через [`_prepare_pool()`](bin/tabpack/tabpack.py:678)
2. Вызывает алгоритм ансамблирования
3. Вычисляет score ансамбля
4. Если улучшилось — обновляет состояние и сбрасывает patience
5. Если нет — уменьшает patience

**Пулы кандидатов по update_type:**
- `final` — только завершённые модели
- `best` — завершённые + лучшие чекпоинты работающих
- `latest` — завершённые + текущие предсказания работающих

---

### [`update_online_ensembles()`](bin/tabpack/tabpack.py:802)

Обновляет все онлайн ансамбли и возвращает отчёты.

---

## Training

### [`generate_training_batches()`](bin/tabpack/tabpack.py:851)

**Назначение:** Генерация случайных батчей для одной эпохи.

**Алгоритм:**
1. Создаёт матрицу случайных значений `(pack_size, train_size)`
2. Сортирует по случайным значениям → порядок батчей
3. Разбивает на батчи заданного размера

---

## Evaluation

### [`_evaluate()`](bin/tabpack/tabpack.py:879)

**Процесс:**
1. Перебирает части (val, test)
2. Для каждой части:
   - Прогоняет все данные батчами
   - Применяет обратную трансформацию (для регрессии)
   - Применяет sigmoid (binclass) или softmax (multiclass)
   - Вычисляет метрики через [`calculate_metrics_pack()`](bin/tabpack/metrics.py:12)

**Возвращает:** `_EvaluateOutput` с metrics, predictions, predictions_torch.

---

## Утилиты

### [`_make_loss_fn_pack()`](bin/tabpack/tabpack.py:1046) — Loss Function

**Назначение:** Создаёт функцию loss для pack с правильной агрегацией.

**Базовые loss функции:**
- Регрессия: `mse_loss`
- Binclass: `binary_cross_entropy_with_logits`
- Multiclass: `cross_entropy`

**Критически важная деталь:**

```python
def loss_fn_pack(y_pred, y_true):
    pack_size = get_pack_size(y_pred)
    losses = base_loss_fn(
        y_pred.flatten(0, 1), y_true.flatten(0, 1), reduction='none'
    )
    losses = losses.unflatten(0, (pack_size, y_true.shape[BATCH_DIM]))
    losses = losses.flatten(BATCH_DIM)
    return losses.mean(BATCH_DIM)  # Среднее по batch, НЕ по pack!
```

**Почему `mean(BATCH_DIM)`, а не `mean()`?**

В training loop используется `loss = losses.sum()`:
```python
losses = loss_fn(apply_model(...), Y_train[batch_idx])  # Форма: (pack_size,)
loss = losses.sum()  # Сумма по pack dimension
```

**Причина:** Градиенты не должны зависеть от количества моделей в pack. Если использовать `.mean()`, то при увеличении `n_models` градиенты уменьшатся, что сломает обучение.

**Влияние:** При изменении `n_models` нужно пропорционально менять `lr` если используется стандартный optimizer (не Pack).

---

### [`_make_optimizer()`](bin/tabpack/tabpack.py:1034)

Фабрика оптимизаторов:
- Стандартные: `torch.optim.*` (AdamW, SGD и др.)
- Кастомные: `AdamWPack`, `MuonAdamWPack`

**Автоматическое удаление pack_size:**
```python
if 'pack_size' not in inspect.signature(optimizer_cls.__init__).parameters:
    kwargs.pop('pack_size', None)
```

Стандартные оптимизаторы не поддерживают `pack_size`, поэтому он автоматически удаляется.

---

### [`_make_online_ensembles()`](bin/tabpack/tabpack.py:963)

Создаёт словарь OnlineEnsemble из конфигурации.

**Обработка algorithm_score_fn:**
```python
algorithm_score_fn = ensemble_config.get('algorithm_score_fn')
if algorithm_score_fn == 'loss':
    ensemble_config['algorithm_score_fn'] = loss_score_fn
```

Если `algorithm_score_fn='loss'`, используется cross-entropy для классификации.

---

## [`main()`](bin/tabpack/tabpack.py:1299) — Полный цикл

### Фаза 1: Инициализация

```
1. _validate_config() — проверка конфига
2. Создание report
3. Установка seed
4. Определение device
5. Настройка AMP (autocast)
6. Загрузка данных:
   - DataPreprocessor.fit_transform()
   - Dataset.to_torch(device)
7. Стандартизация labels (для регрессии)
8. Создание HyperparameterSampler
9. Создание StatePack, FinalStatePack
10. Создание ModelPack
11. Создание OnlineEnsembles
12. Создание Optimizer
13. Создание loss_fn
```

### Фаза 2: Training Loop

**Условие продолжения:**
```python
while (
    report['n_models'] < config['n_models']  # Не достигли цели
    and (timeout is None or timer.elapsed() < timeout)  # Не таймаут
    and (
        online_ensembles is None
        or any(x.is_running for x in online_ensembles.values())
    )  # Ансамбли ещё работают
):
```

**Каждая итерация:**

#### A. Training Phase
```python
for batch_idx in batches:
    losses = loss_fn(apply_model(...), Y_train[batch_idx])
    loss = losses.sum()  # Сумма, не среднее!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1
    state.step()
```

#### B. Evaluation Phase
```python
(eval_metrics, eval_predictions, eval_predictions_torch) = evaluate(...)
state.update(eval_metrics, predictions=..., model_state_dict=...)
```

#### C. Stopping Check
```python
stop_pack_idx = compute_stop_pack_idx(state, ...)
if stop_pack_idx is not None:
    # Оценить лучшие чекпоинты остановленных моделей
    # Добавить в final_state
    # Добавить в experiments
    # Сохранить чекпоинты
    # Удалить из pack
    # Обновить report
```

#### D. Ensemble Update
```python
ensemble_reports, improved = update_online_ensembles(...)
if improved and save_model:
    # Сохранить snapshot ансамбля
```

#### E. Logging
```python
# Вычислить статистику
# Вывести метрики
```

### Фаза 3: Сохранение

**Сохраняемые артефакты:**

1. **checkpoint.pt** — полное состояние для resume:
   ```python
   lib.experiment.dump_checkpoint(exp, {
       'report': report,
       'step': step,
       'random_state': delu.random.get_state(),
       'batch_generator': batch_generator.get_state(),
       'hyperparameter_sampler': hyperparameter_sampler,
       'timer': timer,
   })
   ```

2. **experiments.json** — информация о всех моделях (завершённых и незавершённых)

3. **predictions.npz** — финальные предсказания (если `save_final_predictions=True`)

4. **model.pt** — self-contained модель для инференса (если `save_model=True`):
   ```python
   torch.save({
       'state_dicts': state_dicts_to_save,  # Веса моделей из ансамбля
       'n_models': config['n_models'],
       'model_config': resolved_model_config,  # С per-model списками
       'configs': all_configs,  # Все исходные конфиги
       'n_num_features': dataset.n_num_features,
       'cat_cardinalities': cat_cardinalities,
       'n_classes': n_classes,
       'prediction_type': prediction_type.value,
       'regression_label_stats': regression_label_stats_to_save,
       'data_config': data_config_to_save,
       'ensemble': ensemble_info,  # Информация об ансамбле
       'preprocessor': preprocessor,  # Для воспроизводимости
   }, exp / 'model.pt')
   ```

5. **numlog.npz** — логи обучения (loss, метрики по эпохам)

6. **online_ensemble_history.json** — история обновлений ансамблей (если `track_online_ensemble_history=True`)

---

## Поток configs_T (транспонированные конфигурации)

**Что такое configs_T?**

Когда используется `sampler` или задан `configs`, каждая модель в pack получает свои гиперпараметры. Для удобства обработки эти конфиги транспонируются:

```python
# Исходные конфиги (список словарей)
state.configs = [
    {'model': {'n_blocks': 1, 'dropout': 0.1}, 'optimizer': {'lr': 0.001}},
    {'model': {'n_blocks': 2, 'dropout': 0.2}, 'optimizer': {'lr': 0.002}},
    {'model': {'n_blocks': 3, 'dropout': 0.3}, 'optimizer': {'lr': 0.003}},
]

# Транспонированные (словарь списков)
configs_T = {
    'model': {
        'n_blocks': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
    },
    'optimizer': {
        'lr': [0.001, 0.002, 0.003],
    }
}
```

**Поток в main():**

```python
# 1. Генерация конфигов
state = StatePack(
    pack_size=config['n_models'],
    configs=_prepare_configs(config, hyperparameter_sampler)
)

# 2. Транспонирование
all_configs = deepcopy(state.configs)  # Сохраняем для model.pt
configs_T = transpose_list_of_dicts(state.configs)

# 3. Подготовка model_config
resolved_model_config = _prepare_model_config(
    config,
    configs_T.pop('model')  # Извлекаем и удаляем!
)
# Выводит max_n_blocks, max_d_block, max_d_embedding

# 4. Создание модели
model = ModelPack(
    n_num_features=dataset.n_num_features,
    cat_cardinalities=cat_cardinalities,
    n_classes=n_classes,
    pack_size=state.pack_size,
    **resolved_model_config,  # Включая per-model списки
)

# 5. Создание optimizer
optimizer = _make_optimizer(
    params=lib.deep.make_parameter_groups(model, ...),
    pack_size=model.pack_size,
    **config['optimizer'],  # Базовые параметры
    **(configs_T.pop('optimizer') if 'optimizer' in configs_T else {}),
)

# 6. Проверка пустоты configs_T
assert not configs_T, (
    f'The following fields were not used: {", ".join(configs_T)}'
)
```

**Критически важно:**
- `configs_T` должен быть пуст после всех `pop()`
- `all_configs` сохраняется отдельно для воспроизводимости
- `resolved_model_config` содержит per-model списки (например, `n_blocks = [1, 2, 3, ...]`)
- `max_*` размерности выводятся автоматически из space или `max(value_list)`

---

## Сохранение ensemble snapshot

**Когда сохраняется:**

```python
if first_online_ensemble_improved and save_model:
    # Сохраняем snapshot ансамбля
    ensemble_ids = first_ensemble.ids.tolist()
    ensemble_weights = first_ensemble.weights.tolist()
    ensemble_steps = first_ensemble.steps.tolist()
    
    for idx, eid in enumerate(ensemble_ids):
        estep = ensemble_steps[idx]
        if not checkpoint_store.has_checkpoint(eid, estep):
            # Сохраняем текущие или лучшие веса
            if update_type == 'latest':
                state_dict_to_use = current_model_state_dict  # ТЕКУЩИЕ веса
            else:
                state_dict_to_use = state.best_model_state_dicts  # ЛУЧШИЕ веса
            checkpoint_store.save_checkpoint(eid, estep, state_dict_to_use)
    
    # Удаляем старые чекпоинты не в ансамбле
    for key in checkpoint_store.get_all_checkpoints():
        if key not in current_ensemble_keys:
            checkpoint_store.remove_checkpoint(*key)
```

**Почему это важно:**

Для `update_type='latest'` ансамбль строится на текущих предсказаниях, но веса продолжают меняться. Snapshot необходим для воспроизводимости инференса.

---

## Поток данных в main()

```
Config → _validate_config()
                ↓
         DataPreprocessor.fit_transform(dataset)
                ↓
         Dataset.to_torch(device)
                ↓
         ModelPack + Optimizer
                ↓
    ┌───────────▼───────────┐
    │     Training Loop     │
    │                       │
    │  generate_batches()   │
    │  apply_model()        │
    │  loss.backward()      │
    │  optimizer.step()     │
    │                       │
    │  _evaluate()          │
    │  state.update()       │
    │                       │
    │  compute_stop_idx()   │
    │  pack_remove()        │
    │                       │
    │  update_ensembles()   │
    └───────────┬───────────┘
                ↓
         Сохранение артефактов
```

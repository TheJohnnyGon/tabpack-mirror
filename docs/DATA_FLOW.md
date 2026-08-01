# Поток данных в TabPack

## Обзор

Этот документ описывает, как данные проходят через систему TabPack от конфигурации до финальной модели. Понимание этих потоков критично для внесения изменений и отладки.

---

## 1. Поток configs_T (транспонированные конфигурации)

### Что такое configs_T?

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

### Полный поток

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Загрузка конфига                                              │
│    config = load_config('config.toml')                          │
│    - config['n_models'] = 64                                    │
│    - config['sampler'] = {...} или config['configs'] = [...]   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Генерация конфигов                                            │
│    state = StatePack(                                           │
│        pack_size=config['n_models'],                            │
│        configs=_prepare_configs(config, sampler)                │
│    )                                                            │
│    - Если sampler: configs = [sampler.ask(i) for i in range(n)]│
│    - Если configs: configs = config['configs']                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Транспонирование                                              │
│    all_configs = deepcopy(state.configs)  # Сохраняем для model.pt
│    configs_T = transpose_list_of_dicts(state.configs)           │
│    - configs_T['model']['n_blocks'] = [1, 2, 3, ...]           │
│    - configs_T['optimizer']['lr'] = [0.001, 0.002, ...]        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Подготовка model_config                                       │
│    resolved_model_config = _prepare_model_config(               │
│        config,                                                  │
│        configs_T.pop('model')  # Извлекаем и удаляем!          │
│    )                                                            │
│    - Выводит max_n_blocks, max_d_block, max_d_embedding        │
│    - Объединяет per-model параметры в model_config              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Создание модели                                               │
│    model = ModelPack(                                           │
│        n_num_features=dataset.n_num_features,                   │
│        cat_cardinalities=cat_cardinalities,                     │
│        n_classes=n_classes,                                     │
│        pack_size=state.pack_size,                               │
│        **resolved_model_config,  # Включая per-model списки    │
│    )                                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Создание optimizer                                            │
│    optimizer = _make_optimizer(                                 │
│        params=lib.deep.make_parameter_groups(model, ...),       │
│        pack_size=model.pack_size,                               │
│        **config['optimizer'],  # Базовые параметры             │
│        **(configs_T.pop('optimizer') if 'optimizer' in configs_T else {}),
│    )                                                            │
│    - Per-model lr, weight_decay передаются как списки          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Проверка пустоты configs_T                                    │
│    assert not configs_T, (                                      │
│        f'The following fields were not used: {", ".join(configs_T)}'
│    )                                                            │
│    - Если остались неиспользованные поля — ошибка!             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. Сохранение в model.pt                                         │
│    torch.save({                                                 │
│        'state_dicts': state_dicts_to_save,                      │
│        'n_models': config['n_models'],                          │
│        'model_config': resolved_model_config,  # С per-model   │
│        'configs': all_configs,  # Все исходные конфиги         │
│        ...                                                      │
│    }, exp / 'model.pt')                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые моменты

1. **configs_T должен быть пуст** — все поля должны быть использованы через `pop()`
2. **all_configs сохраняется отдельно** — для воспроизводимости и анализа
3. **resolved_model_config содержит per-model списки** — например, `n_blocks = [1, 2, 3, ...]`
4. **max размерности выводятся автоматически** — из space или max(value_list)

### Подводные камни

#### Проблема: Неиспользованные поля в configs_T

```python
# Если sampler генерирует поле, которое не используется:
configs_T = {
    'model': {'n_blocks': [1, 2, 3]},
    'optimizer': {'lr': [0.001, 0.002, 0.003]},
    'custom_field': [1, 2, 3],  # Не используется нигде!
}

# Assert в строке 1457:
assert not configs_T, 'The following fields were not used: custom_field'
```

**Решение:** Либо использовать поле, либо не генерировать его в sampler.

#### Проблема: Неправильный max размер

```python
# Если n_blocks задан как список:
model_config['n_blocks'] = [1, 2, 3]
# max_n_blocks выводится автоматически:
model_config['max_n_blocks'] = max([1, 2, 3])  # = 3

# Но если n_blocks задан как скаляр:
model_config['n_blocks'] = 2
# max_n_blocks должен быть None (все модели одинаковые):
model_config['max_n_blocks'] = None
```

**Решение:** `_prepare_model_config()` обрабатывает оба случая.

---

## 2. Поток prediction_type

### Что такое prediction_type?

```python
prediction_type = (
    PredictionType.LABELS if dataset.task.is_regression else PredictionType.PROBS
)
```

**Важно:** Предсказания сохраняются в "aggregation-friendly" формате:
- **Регрессия** — `LABELS` (денормализованные значения)
- **Классификация** — `PROBS` (вероятности после sigmoid/softmax)

### Полный поток

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Определение prediction_type                                   │
│    prediction_type = (                                          │
│        PredictionType.LABELS if dataset.task.is_regression      │
│        else PredictionType.PROBS                                │
│    )                                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Создание apply_model с autocast                               │
│    apply_model = apply_model_impl if autocast is None           │
│                  else autocast(apply_model_impl)                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Evaluation (_evaluate)                                        │
│    def _evaluate(apply_model, model, optimizer, dataset, ...):  │
│        for part in parts:                                       │
│            for batch_idx in batches:                            │
│                y_pred = apply_model(model, dataset, part, batch_idx)
│                # y_pred — это LOGITS (сырой выход модели)       │
│                                                                 │
│            # Постобработка:                                     │
│            if task.is_regression:                               │
│                # Денормализация: y_pred * std + mean            │
│                y_pred = denormalize(y_pred, regression_label_stats)
│            elif task.is_binclass:                               │
│                y_pred = torch.sigmoid(y_pred)  # → PROBS       │
│            elif task.is_multiclass:                             │
│                y_pred = torch.softmax(y_pred, dim=-1)  # → PROBS│
│                                                                 │
│            # Сохранение:                                        │
│            predictions[part] = y_pred.numpy()  # LABELS или PROBS
│            predictions_torch[part] = y_pred                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Сохранение в StatePack                                        │
│    state.update(                                                │
│        eval_metrics,                                            │
│        predictions=eval_predictions,  # LABELS или PROBS       │
│        predictions_torch=eval_predictions_torch,                │
│        model_state_dict=model.state_dict(),                     │
│    )                                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Использование в OnlineEnsemble                                │
│    online_ensemble = OnlineEnsemble(                            │
│        prediction_type=prediction_type,  # Передается явно!    │
│        ...                                                      │
│    )                                                            │
│    - Ансамбль работает с LABELS/PROBS, не с LOGITS             │
│    - Усреднение происходит в правильном пространстве            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Сохранение в model.pt                                         │
│    torch.save({                                                 │
│        ...                                                      │
│        'prediction_type': prediction_type.value,  # 'labels' или 'probs'
│        'regression_label_stats': regression_label_stats_to_save,
│        ...                                                      │
│    }, exp / 'model.pt')                                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Инференс (infer.py)                                           │
│    artifact = load_model_artifact('model.pt')                   │
│    prediction_type = artifact['prediction_type']                │
│    regression_label_stats = artifact['regression_label_stats']  │
│                                                                 │
│    # evaluate_ensemble использует prediction_type               │
│    # для правильной интерпретации предсказаний                  │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые моменты

1. **LOGITS никогда не сохраняются** — только LABELS или PROBS
2. **Денормализация происходит в `_evaluate`** — для регрессии
3. **prediction_type передается в OnlineEnsemble** — для правильного усреднения
4. **regression_label_stats сохраняется в model.pt** — для денормализации при инференсе

### Подводные камни

#### Проблема: Сохранение logits вместо probs

```python
# НЕПРАВИЛЬНО:
y_pred = model(x_num, x_cat)  # Это LOGITS!
predictions[part] = y_pred.numpy()

# ПРАВИЛЬНО:
if task.is_binclass:
    y_pred = torch.sigmoid(y_pred)  # → PROBS
predictions[part] = y_pred.numpy()
```

**Почему важно:** Ансамбль усредняет предсказания. Усреднение logits не имеет смысла (нужно усреднять probs).

#### Проблема: Забытая денормализация

```python
# Для регрессии labels стандартизуются:
dataset.try_standardize_labels_()  # y = (y - mean) / std

# Модель обучается предсказывать стандартизованные labels
# При evaluation нужно денормализовать:
y_pred = y_pred * std + mean  # Иначе метрики будут неправильные!
```

**Решение:** `_evaluate()` автоматически денормализует через `regression_label_stats`.

---

## 3. Поток regression_label_stats

### Что такое regression_label_stats?

Для регрессии labels стандартизуются для улучшения обучения:

```python
@dataclass
class RegressionLabelStats:
    mean: float
    std: float
```

### Полный поток

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Стандартизация labels                                         │
│    regression_label_stats = dataset.try_standardize_labels_()   │
│    - Если регрессия: y = (y - mean) / std                      │
│    - Если классификация: None                                   │
│    - Возвращает RegressionLabelStats(mean, std)                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Обучение на стандартизованных labels                          │
│    loss_fn = _make_loss_fn_pack(task_type)                      │
│    losses = loss_fn(model(x), Y_train)  # Y_train стандартизован│
│    - Модель учится предсказывать (y - mean) / std              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Денормализация при evaluation                                 │
│    def _evaluate(..., regression_label_stats, ...):             │
│        y_pred = apply_model(model, dataset, part, batch_idx)    │
│        if regression_label_stats is not None:                   │
│            y_pred = y_pred * std + mean  # Денормализация!     │
│        predictions[part] = y_pred.numpy()                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Сохранение в model.pt                                         │
│    regression_label_stats_to_save = (                           │
│        None if regression_label_stats is None                   │
│        else dataclasses.asdict(regression_label_stats)          │
│    )                                                            │
│    torch.save({                                                 │
│        ...                                                      │
│        'regression_label_stats': regression_label_stats_to_save,
│        ...                                                      │
│    }, exp / 'model.pt')                                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Инференс (infer.py)                                           │
│    artifact = load_model_artifact('model.pt')                   │
│    regression_label_stats = artifact['regression_label_stats']  │
│                                                                 │
│    # evaluate_ensemble денормализует предсказания:              │
│    if regression_label_stats is not None:                       │
│        y_pred = y_pred * std + mean                             │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые моменты

1. **Стандартизация улучшает обучение** — градиенты более стабильные
2. **Денормализация происходит в `_evaluate`** — автоматически
3. **stats сохраняется в model.pt** — для воспроизводимости инференса
4. **Для классификации stats = None** — не нужно

### Подводные камни

#### Проблема: Забытая денормализация при инференсе

```python
# Если забыть денормализовать:
y_pred = model(x)  # Это стандартизованное предсказание!
metrics = calculate_metrics(y_true, y_pred)  # Неправильные метрики!

# ПРАВИЛЬНО:
if regression_label_stats is not None:
    y_pred = y_pred * std + mean
metrics = calculate_metrics(y_true, y_pred)  # Правильные метрики!
```

#### Проблема: Неправильный порядок операций

```python
# НЕПРАВИЛЬНО: денормализация ДО sigmoid
y_pred = model(x)
y_pred = y_pred * std + mean  # Для регрессии OK
y_pred = torch.sigmoid(y_pred)  # Для классификации — НЕПРАВИЛЬНО!

# ПРАВИЛЬНО: денормализация только для регрессии
if task.is_regression:
    y_pred = y_pred * std + mean
elif task.is_binclass:
    y_pred = torch.sigmoid(y_pred)
```

---

## 4. Поток model.pt (сохранение и загрузка)

### Формат model.pt

```python
{
    'state_dicts': {                    # Веса моделей
        (model_id, step): state_dict,   # Новый формат
    },
    'n_models': int,                    # Общее количество моделей
    'model_config': dict,               # Конфигурация модели (с per-model списками)
    'configs': list,                    # Все исходные конфиги
    'n_num_features': int,              # Количество числовых признаков
    'cat_cardinalities': list,          # Кардинальности категорий
    'n_classes': int,                   # Количество классов (None для регрессии)
    'prediction_type': str,             # 'labels' или 'probs'
    'regression_label_stats': dict,     # Статистика для денормализации
    'data_config': dict,               # Конфигурация данных (с seed)
    'ensemble': dict,                   # Информация об ансамбле
    'preprocessor': DataPreprocessor,   # Сохранённый препроцессор
}
```

### Полный поток сохранения

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Инициализация checkpoint_store                                │
│    save_model = config.get('save_model', False)                 │
│    checkpoint_store = EnsembleCheckpointStore()                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Сохранение чекпоинтов при остановке моделей                   │
│    if save_model:                                               │
│        for i in stop_pack_idx:                                  │
│            member_id = state.ids[i]                             │
│            member_step = state.best_steps[i]                    │
│            checkpoint_store.save_checkpoint(                    │
│                model_id=member_id,                              │
│                step=member_step,                                │
│                state_dict={                                     │
│                    name: value[i:i+1]  # Pack dimension = 1    │
│                    for name, value in state.best_model_state_dicts.items()
│                },                                               │
│            )                                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Сохранение ensemble snapshot при улучшении                    │
│    if first_online_ensemble_improved and save_model:            │
│        ensemble_ids = first_ensemble.ids.tolist()               │
│        ensemble_weights = first_ensemble.weights.tolist()       │
│        ensemble_steps = first_ensemble.steps.tolist()           │
│                                                                 │
│        for idx, eid in enumerate(ensemble_ids):                 │
│            estep = ensemble_steps[idx]                          │
│            if not checkpoint_store.has_checkpoint(eid, estep):  │
│                # Сохраняем текущие или лучшие веса              │
│                if update_type == 'latest':                      │
│                    state_dict = current_model_state_dict        │
│                else:                                            │
│                    state_dict = state.best_model_state_dicts    │
│                checkpoint_store.save_checkpoint(eid, estep, ...)│
│                                                                 │
│        # Удаляем старые чекпоинты не в ансамбле                 │
│        for key in checkpoint_store.get_all_checkpoints():       │
│            if key not in current_ensemble_keys:                 │
│                checkpoint_store.remove_checkpoint(*key)         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Формирование state_dicts для model.pt                         │
│    if last_ensemble_snapshot is not None:                       │
│        ensemble_info = {                                        │
│            'ids': last_ensemble_snapshot['ids'],                │
│            'weights': last_ensemble_snapshot['weights'],        │
│            'steps': last_ensemble_snapshot['steps'],            │
│        }                                                        │
│        state_dicts_to_save = {}                                 │
│        for idx, eid in enumerate(ensemble_info['ids']):         │
│            estep = ensemble_info['steps'][idx]                  │
│            state_dicts_to_save[(eid, estep)] = (                │
│                checkpoint_store.get_checkpoint(eid, estep)      │
│            )                                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Сохранение model.pt                                           │
│    torch.save({                                                 │
│        'state_dicts': state_dicts_to_save,                      │
│        'n_models': config['n_models'],                          │
│        'model_config': resolved_model_config,                   │
│        'configs': all_configs,                                  │
│        'n_num_features': dataset.n_num_features,                │
│        'cat_cardinalities': cat_cardinalities,                  │
│        'n_classes': n_classes,                                  │
│        'prediction_type': prediction_type.value,                │
│        'regression_label_stats': regression_label_stats_to_save,│
│        'data_config': data_config_to_save,                      │
│        'ensemble': ensemble_info,                               │
│        'preprocessor': preprocessor,                            │
│    }, exp / 'model.pt')                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Полный поток загрузки (инференс)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Загрузка артефакта                                            │
│    artifact = load_model_artifact('model.pt')                   │
│    state_dicts = artifact['state_dicts']                        │
│    ensemble_info = artifact['ensemble']                         │
│    preprocessor = artifact['preprocessor']                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Определение формата state_dicts                               │
│    # Новый формат: ключи — tuple (model_id, step)              │
│    # Legacy формат: ключи — int (model_id)                     │
│    first_key = next(iter(state_dicts.keys()))                   │
│    is_new_format = isinstance(first_key, tuple)                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Определение model_ids для ансамбля                            │
│    if ensemble_info is not None:                                │
│        model_ids = ensemble_info['ids']                         │
│        weights = ensemble_info['weights']                       │
│        steps = ensemble_info['steps']                           │
│    else:                                                        │
│        # Fallback: все сохранённые модели                       │
│        model_ids = list(state_dicts.keys())                     │
│        weights = None                                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Построение модели                                             │
│    model = build_model_for_ensemble(artifact, model_ids)        │
│    - Извлекает per-model параметры из model_config              │
│    - Создает ModelPack с правильным pack_size                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Загрузка весов                                                │
│    load_all_weights(model, state_dicts, model_ids, steps)       │
│    - Загружает веса в правильные позиции pack                   │
│    - Поддерживает оба формата (legacy и новый)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Оценка ансамбля                                               │
│    evaluate_ensemble(                                           │
│        model, dataset, model_ids, weights,                      │
│        regression_label_stats=artifact['regression_label_stats'],
│        prediction_type=artifact['prediction_type'],             │
│    )                                                            │
│    - Прогоняет данные через все модели                          │
│    - Усредняет предсказания (с весами если есть)                │
│    - Денормализует для регрессии                                │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые моменты

1. **state_dicts использует tuple (model_id, step)** — новый формат
2. **ensemble_info определяет какие модели использовать** — не все сохранённые
3. **preprocessor сохраняется для воспроизводимости** — не нужно заново обучать
4. **checkpoint_store удаляет старые чекпоинты** — предотвращает утечку памяти

### Подводные камни

#### Проблема: Legacy формат без ensemble_info

```python
# Legacy model.pt:
{
    'state_dicts': {0: {...}, 1: {...}, 2: {...}},  # Все модели
    'ensemble': None,  # Нет информации об ансамбле
}

# Инференс использует ВСЕ модели (простое среднее)
# Это может быть не оптимально!
```

**Решение:** Всегда использовать `save_model=True` с `online_ensembles`.

#### Проблема: Неправильный update_type

```python
# Для update_type='latest':
# Ансамбль строится на ТЕКУЩИХ предсказаниях
# Но веса продолжают меняться!
# Нужно сохранить snapshot весов в момент улучшения ансамбля

if update_type == 'latest':
    state_dict_to_use = current_model_state_dict  # Текущие веса
else:
    state_dict_to_use = state.best_model_state_dicts  # Лучшие веса
```

**Решение:** Ensemble snapshot сохраняется при каждом улучшении ансамбля.

---

## 5. Поток данных через Dataset и DataPreprocessor

### Полный поток

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Загрузка сырых данных                                         │
│    dataset = Dataset.from_dir(path, split_id)                   │
│    - Загружает x_num.npy, x_cat.npy, y.npy                     │
│    - Применяет сплит (train/val/test)                           │
│    - Возвращает Dataset[np.ndarray]                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Создание препроцессора                                        │
│    data_config_with_seed = {**config['data'], 'seed': config['seed']}
│    preprocessor = DataPreprocessor(data_config_with_seed)       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Fit + Transform                                               │
│    dataset = preprocessor.fit_transform(dataset)                │
│    - fit(): обучает трансформеры на train                       │
│    - transform(): применяет ко всем частям                      │
│    - Сохраняет индексы признаков и трансформеры                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Стандартизация labels (для регрессии)                         │
│    regression_label_stats = dataset.try_standardize_labels_()   │
│    - Если регрессия: y = (y - mean) / std                      │
│    - Возвращает RegressionLabelStats                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Конвертация в torch tensors                                   │
│    dataset = dataset.to_torch(device)                           │
│    - Перемещает данные на GPU                                   │
│    - Освобождает numpy массивы                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Сохранение preprocessor в model.pt                            │
│    torch.save({                                                 │
│        ...                                                      │
│        'preprocessor': preprocessor,                            │
│        'data_config': data_config_with_seed,                    │
│        ...                                                      │
│    }, exp / 'model.pt')                                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Инференс: использование сохранённого preprocessor             │
│    artifact = load_model_artifact('model.pt')                   │
│    preprocessor = artifact['preprocessor']                      │
│                                                                 │
│    # Загрузка новых данных:                                     │
│    new_dataset = Dataset.from_dir(new_path, split_id)           │
│    new_dataset = preprocessor.transform(new_dataset)  # Без fit!│
│    - Применяет те же трансформеры                               │
│    - Гарантирует идентичный препроцессинг                       │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые моменты

1. **fit() только на train** — предотвращает data leakage
2. **transform() применяется ко всем частям** — train, val, test
3. **preprocessor сохраняется в model.pt** — для воспроизводимости
4. **При инференсе используется transform() без fit()** — те же трансформеры

---

## 6. Диаграмма полного потока данных

```
config.toml
    ↓
┌──────────────────────────────────────────────────────────────┐
│ main()                                                        │
│                                                               │
│  config['seed'] ─────────────────────────────────────────┐   │
│  config['data'] ─────────────────────────────────────┐   │   │
│  config['n_models'] ─────────────────────────────┐   │   │   │
│  config['sampler'] ──────────────────────────┐   │   │   │   │
│                                              ↓   ↓   ↓   ↓   │
│                                    ┌──────────────────────┐  │
│                                    │ HyperparameterSampler│  │
│                                    └──────────────────────┘  │
│                                              ↓               │
│                                    ┌──────────────────────┐  │
│                                    │ state.configs        │  │
│                                    │ (list of dicts)      │  │
│                                    └──────────────────────┘  │
│                                              ↓               │
│                                    ┌──────────────────────┐  │
│                                    │ configs_T            │  │
│                                    │ (transposed)         │  │
│                                    └──────────────────────┘  │
│                                         ↓         ↓          │
│                              ┌────────────┐  ┌────────────┐  │
│                              │ model_config│  │ optimizer  │  │
│                              └────────────┘  └────────────┘  │
│                                    ↓              ↓          │
│                              ┌────────────────────────────┐  │
│                              │ ModelPack + Optimizer      │  │
│                              └────────────────────────────┘  │
│                                              ↓               │
│                                    ┌──────────────────────┐  │
│                                    │ Training Loop        │  │
│                                    │ - loss_fn            │  │
│                                    │ - evaluate           │  │
│                                    │ - state.update       │  │
│                                    │ - online_ensembles   │  │
│                                    └──────────────────────┘  │
│                                              ↓               │
│                                    ┌──────────────────────┐  │
│                                    │ checkpoint_store     │  │
│                                    │ (best weights)       │  │
│                                    └──────────────────────┘  │
│                                              ↓               │
│                                    ┌──────────────────────┐  │
│                                    │ model.pt             │  │
│                                    │ - state_dicts        │  │
│                                    │ - ensemble_info      │  │
│                                    │ - preprocessor       │  │
│                                    │ - regression_stats   │  │
│                                    └──────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                                              ↓
┌──────────────────────────────────────────────────────────────┐
│ infer.py                                                      │
│                                                               │
│  load_model_artifact('model.pt')                             │
│  build_model_for_ensemble(artifact, model_ids)               │
│  load_all_weights(model, state_dicts)                        │
│  evaluate_ensemble(model, dataset, weights, ...)             │
│                                                               │
│  Результат: метрики ансамбля                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Чеклист для отладки

### Проблема: Неправильные метрики при инференсе

**Проверьте:**
1. `prediction_type` в model.pt — 'labels' или 'probs'?
2. `regression_label_stats` — денормализуются ли предсказания?
3. `ensemble_info` — используются ли правильные модели?
4. `preprocessor` — применяется ли тот же препроцессинг?

### Проблема: Ошибка "configs_T not empty"

**Причина:** Sampler генерирует поля, которые не используются.

**Решение:**
1. Проверьте что все поля из `configs_T` используются через `pop()`
2. Либо удалите поле из sampler space
3. Либо добавьте использование в `_prepare_model_config()` или `_make_optimizer()`

### Проблема: OOM при увеличении n_models

**Причина:** Pack тензоры имеют форму `(pack_size, batch_size, features)`.

**Решение:**
1. Уменьшите `batch_size` пропорционально `n_models`
2. Используйте `amp_dtype='bfloat16'` для уменьшения VRAM
3. Уменьшите `max_n_blocks` или `max_d_block`

### Проблема: Ансамбль не улучшается

**Проверьте:**
1. `online_ensembles[*].patience` — достаточно ли терпения?
2. `online_ensembles[*].update_type` — 'latest' может быть нестабильным
3. `online_ensembles[*].options.max_ensemble_size` — достаточно ли большой?
4. Разнообразие моделей — разные ли гиперпараметры у моделей?

---

## Ссылки

- [`_prepare_configs()`](../bin/tabpack/tabpack.py:1190)
- [`_prepare_model_config()`](../bin/tabpack/tabpack.py:1204)
- [`_evaluate()`](../bin/tabpack/tabpack.py:879)
- [`OnlineEnsemble`](../bin/tabpack/tabpack.py:603)
- [`EnsembleCheckpointStore`](../bin/tabpack/ensemble_checkpoint.py:19)
- [`DataPreprocessor`](../lib/data.py:54)
- [`build_model_for_ensemble()`](../infer/infer.py:83)
- [`evaluate_ensemble()`](../infer/infer.py:139)

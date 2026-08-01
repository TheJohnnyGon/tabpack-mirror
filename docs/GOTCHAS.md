# Подводные камни и неявные зависимости TabPack

## Обзор

Этот документ описывает критические подводные камни, неявные зависимости и вещи, которые **НЕЛЬЗЯ** делать при внесении изменений в TabPack.

---

## 0. Расхождение метрик между report.json и infer.py (bfloat16 vs float32)

### Проблема

При сравнении метрик из `report.json` (online ensemble) и `infer.py` наблюдаются небольшие различия:

```
=== Ensemble Results (infer.py) ===
  [val]
    accuracy:  0.7960
    roc-auc:   0.8662

=== Nirvana Results (report.json) ===
  [val]
    accuracy:  0.7961
    roc-auc:   0.8661
```

### Причина

**bfloat16 AMP при обучении vs float32 при инференсе.**

**При обучении** ([`tabpack.py:1410`](../bin/tabpack/tabpack.py:1410)):
```python
apply_model = apply_model_impl if autocast is None else autocast(apply_model_impl)
```

С `amp_dtype = "bfloat16"` forward pass выполняется в **bfloat16**:
- 8 бит экспоненты (как float32)
- **7 бит мантиссы** (vs 23 у float32)
- Точность ~3 десятичных знака

**При инференсе** ([`infer.py:176`](../infer/infer.py:176)):
```python
with torch.inference_mode():
    eval_result = _evaluate(apply_model_impl, model, ...)
```

**Без autocast** — модель работает в **float32** (полная точность).

### Влияние на метрики

```
bfloat16: 0.7960 (точность ~3 знака)
float32:  0.7961 (точность ~7 знаков)
```

Расхождение в 4-м знаке после запятой — **типично для bfloat16** и является **ожидаемым поведением**, а не багом.

### Почему это нормально

1. **bfloat16 используется для ускорения обучения** — меньше памяти, быстрее вычисления
2. **float32 используется для инференса** — максимальная точность предсказаний
3. **Расхождение < 0.001** — приемлемо для production использования
4. **Метрики в report.json** — отражают качество модели при обучении (с AMP)
5. **Метрики в infer.py** — отражают качество модели при инференсе (без AMP)

### Когда это влияет

- При сравнении метрик из `report.json` и `infer.py`
- При отладке расхождений в метриках
- При принятии решения о качестве модели

### Рекомендации

1. **Используйте метрики из infer.py** для финальной оценки модели (float32 точнее)
2. **Метрики из report.json** — для мониторинга во время обучения
3. **Расхождение < 0.001** — нормально, не требует исправления
4. **Расхождение > 0.01** — проверьте конфигурацию AMP и модель

### Альтернатива: использовать bfloat16 при инференсе

Если нужно точно воспроизвести метрики из report.json:

```python
# infer.py
with torch.inference_mode():
    with torch.autocast(device.type, dtype=torch.bfloat16):
        eval_result = _evaluate(apply_model_impl, model, ...)
```

**Но это не рекомендуется** — float32 даёт более точные предсказания.

---

## 1. Loss scaling: sum() vs mean()

### Проблема

В training loop используется `loss.sum()`, а не `loss.mean()`:

```python
# bin/tabpack/tabpack.py:1594
losses = loss_fn(apply_model(...), Y_train[batch_idx])
loss = losses.sum()  # Сумма по pack dimension!
```

### Почему это важно

`_make_loss_fn_pack()` возвращает тензор формы `(pack_size,)` — по одному loss на каждую модель:

```python
def loss_fn_pack(y_pred, y_true):
    pack_size = get_pack_size(y_pred)
    losses = base_loss_fn(
        y_pred.flatten(0, 1), y_true.flatten(0, 1), reduction='none'
    )
    losses = losses.unflatten(0, (pack_size, y_true.shape[BATCH_DIM]))
    losses = losses.flatten(BATCH_DIM)
    return losses.mean(BATCH_DIM)  # Среднее по batch, не по pack!
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО:
loss = losses.mean()  # Градиенты уменьшатся в pack_size раз!
```

### Последствия

Если заменить `sum()` на `mean()`:
- Градиенты уменьшатся в `pack_size` раз
- При увеличении `n_models` обучение станет медленнее
- Learning rate нужно будет масштабировать пропорционально `n_models`

### Когда это влияет

- При добавлении нового loss — убедитесь что `loss_fn_pack` возвращает `(pack_size,)`
- При изменении training loop — сохраняйте `loss.sum()`
- При использовании стандартного optimizer (не Pack) — нужно масштабировать lr

---

## 2. PACK_DIM = 0, BATCH_DIM = 1

### Проблема

Все тензоры в pack-модулях имеют форму `(pack_size, batch_size, features)`:

```python
# bin/tabpack/nn.py:19-20
PACK_DIM = 0
BATCH_DIM = 1
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО:
x = x.transpose(0, 1)  # Меняет местами pack и batch!
x = x.permute(1, 0, 2)  # То же самое

# НЕПРАВИЛЬНО:
x = x.mean(dim=0)  # Усредняет по pack, не по batch!
```

### Где это критично

1. **LinearPack** — веса имеют форму `(pack_size, out_features, in_features)`
2. **MLPBackbonePack** — `_compute_block_idx_list()` использует pack dimension
3. **DropoutPack** — маски генерируются с учётом pack dimension
4. **module_pack_load_state_dict** — индексы относятся к pack dimension
5. **_evaluate** — предсказания имеют форму `(pack_size, batch_size, ...)`

### Последствия

Если перепутать pack и batch:
- Модели будут использовать чужие веса
- Метрики будут неправильные
- Ансамбль усреднит не те предсказания

---

## 3. Координация pack_remove

### Проблема

При удалении моделей из pack нужно синхронно обновить **три** компонента:

```python
# bin/tabpack/tabpack.py:505-516
def pack_remove(model, optimizer, state, *, pack_idx):
    pack_idx_torch = torch.as_tensor(pack_idx, device=...)
    old_to_new = bin.tabpack.nn.module_pack_remove(model, pack_idx_torch)
    bin.tabpack.optim.optimizer_pack_remove(optimizer, pack_idx_torch, old_to_new)
    state.remove(pack_idx)
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: удалить только из модели
bin.tabpack.nn.module_pack_remove(model, pack_idx)
# Optimizer и state всё ещё ссылаются на старые параметры!

# НЕПРАВИЛЬНО: удалить в неправильном порядке
state.remove(pack_idx)  # state обновился
bin.tabpack.nn.module_pack_remove(model, pack_idx)  # Но old_to_new не передан в optimizer!
```

### Последствия

Если не синхронизировать:
- Optimizer будет обновлять несуществующие параметры
- State будет иметь неправильный pack_size
- `pack_validate()` упадёт с assert

### Проверка

После любого удаления вызывайте:
```python
pack_validate(model, optimizer, state)
```

---

## 4. configs_T должен быть пуст

### Проблема

После извлечения per-model параметров `configs_T` должен быть пуст:

```python
# bin/tabpack/tabpack.py:1457-1460
assert not configs_T, (
    'The following fields of the generated configs were not used:'
    f' {", ".join(configs_T)}'
)
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: добавить поле в sampler space без использования
config['sampler'] = {
    'space': {
        'model': {'n_blocks': ['_tune_', 'int', 1, 3]},
        'custom_field': ['_tune_', 'uniform', 0.0, 1.0],  # Не используется!
    }
}
# Assert упадёт: "The following fields were not used: custom_field"
```

### Как исправить

1. **Использовать поле** — добавить обработку в `_prepare_model_config()` или `_make_optimizer()`
2. **Не генерировать** — убрать из sampler space
3. **Явно удалить** — `configs_T.pop('custom_field')` перед assert

### Когда это влияет

- При добавлении новых per-model гиперпараметров
- При изменении sampler space
- При создании кастомных конфигов

---

## 5. prediction_type: LABELS vs PROBS vs LOGITS

### Проблема

Предсказания сохраняются в "aggregation-friendly" формате:

```python
# bin/tabpack/tabpack.py:1407-1409
prediction_type = (
    PredictionType.LABELS if dataset.task.is_regression else PredictionType.PROBS
)
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: сохранить logits
y_pred = model(x_num, x_cat)  # Это LOGITS!
predictions[part] = y_pred.numpy()

# НЕПРАВИЛЬНО: усреднять logits в ансамбле
ensemble_pred = logits.mean(dim=0)  # Не имеет смысла!
```

### Правильный поток

```python
# В _evaluate():
y_pred = apply_model(model, dataset, part, batch_idx)  # LOGITS

if task.is_regression:
    y_pred = y_pred * std + mean  # → LABELS
elif task.is_binclass:
    y_pred = torch.sigmoid(y_pred)  # → PROBS
elif task.is_multiclass:
    y_pred = torch.softmax(y_pred, dim=-1)  # → PROBS

predictions[part] = y_pred.numpy()  # LABELS или PROBS
```

### Последствия

Если сохранить logits:
- Ансамбль усреднит logits (неправильно для классификации)
- Метрики будут неправильные
- Инференс даст некорректные результаты

---

## 6. regression_label_stats: денормализация

### Проблема

Для регрессии labels стандартизуются:

```python
# bin/tabpack/tabpack.py:1344
regression_label_stats = dataset.try_standardize_labels_()
# y = (y - mean) / std
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: забыть денормализовать
y_pred = model(x)  # Это стандартизованное предсказание!
metrics = calculate_metrics(y_true, y_pred)  # Неправильные метрики!

# НЕПРАВИЛЬНО: денормализовать для классификации
if task.is_binclass:
    y_pred = y_pred * std + mean  # Для классификации stats = None!
```

### Правильный поток

```python
# В _evaluate():
if regression_label_stats is not None:  # Только для регрессии!
    y_pred *= regression_label_stats.std
    y_pred += regression_label_stats.mean
```

### Где это критично

1. **_evaluate()** — денормализует предсказания
2. **model.pt** — сохраняет `regression_label_stats`
3. **infer.py** — загружает и использует для денормализации

### Последствия

Если забыть денормализовать:
- Метрики будут в стандартизованном пространстве (неинтерпретируемые)
- Предсказания при инференсе будут неправильные

---

## 7. Online ensemble patience

### Проблема

Training loop продолжается пока хотя бы один ансамбль `is_running`:

```python
# bin/tabpack/tabpack.py:1558-1564
while (
    report['n_models'] < config['n_models']
    and (timeout is None or timer.elapsed() < timeout)
    and (
        online_ensembles is None
        or any(x.is_running for x in online_ensembles.values())
    )
):
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: установить patience=0
config['online_ensembles'] = {
    'greedy': {
        'type': 'greedy',
        'patience': 0,  # Ансамбль остановится сразу!
    }
}
```

### Как работает patience

```python
# OnlineEnsemble:
self._remaining_patience = patience

def update(...):
    if improved:
        self._remaining_patience = self._patience  # Сброс
    else:
        self._remaining_patience -= 1  # Уменьшение

@property
def is_running(self):
    return self._remaining_patience >= 0
```

### Последствия

Если все ансамбли исчерпали patience:
- Training loop останавливается
- Даже если `n_models` не достигнут
- Даже если `timeout` не истёк

### Рекомендация

- Для стабильного обучения: `patience >= 16`
- Для быстрого прототипирования: `patience >= 4`
- Для production: `patience >= 32`

---

## 8. save_model и ensemble snapshot

### Проблема

Для `update_type='latest'` ансамбль строится на текущих предсказаниях, но веса продолжают меняться:

```python
# bin/tabpack/tabpack.py:1803-1858
if first_online_ensemble_improved and save_model:
    # Сохраняем snapshot весов в момент улучшения ансамбля
    for idx, eid in enumerate(ensemble_ids):
        estep = ensemble_steps[idx]
        if not checkpoint_store.has_checkpoint(eid, estep):
            if update_type == 'latest':
                state_dict_to_use = current_model_state_dict  # ТЕКУЩИЕ веса
            else:
                state_dict_to_use = state.best_model_state_dicts  # ЛУЧШИЕ веса
            checkpoint_store.save_checkpoint(eid, estep, state_dict_to_use)
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: save_model=False с online_ensembles
config['save_model'] = False
config['online_ensembles'] = {
    'greedy': {
        'type': 'greedy',
        'update_type': 'latest',  # Веса не сохранятся!
    }
}
# Инференс не сможет воспроизвести ансамбль!
```

### Последствия

Если `save_model=False`:
- `model.pt` не создаётся
- Ensemble snapshot не сохраняется
- Инференс невозможен
- Все веса теряются после обучения

### Рекомендация

Всегда использовать `save_model=True` с `online_ensembles`.

---

## 9. n_epochs = -1 означает бесконечность

### Проблема

```python
# bin/tabpack/tabpack.py:519-534
def compute_stop_pack_idx(state, epoch_size, n_epochs, patience):
    stop_pack_idx = []
    for i in range(state.pack_size):
        # Patience check
        if state.n_consequtive_bad_updates[i] > patience:
            stop_pack_idx.append(i)
        # Epoch limit check
        elif n_epochs >= 0 and state.steps[i] // epoch_size >= n_epochs:
            stop_pack_idx.append(i)
    return np.array(stop_pack_idx) if stop_pack_idx else None
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: n_epochs=-1 без patience
config['n_epochs'] = -1
config['patience'] = 0  # Модели остановятся сразу!

# НЕПРАВИЛЬНО: n_epochs=-1 без timeout и без online_ensembles
config['n_models'] = 64
config['n_epochs'] = -1
config['timeout'] = None
config['online_ensembles'] = None
# Обучение будет идти бесконечно пока все 64 модели не остановятся по patience!
```

### Рекомендация

- `n_epochs=-1` + `patience=16` — стандартный вариант
- `n_epochs=-1` + `timeout=3600` — ограничение по времени
- `n_epochs=100` + `patience=16` — двойное ограничение

---

## 10. track_memory_usage останавливает обучение

### Проблема

```python
# bin/tabpack/tabpack.py:1619-1621
if config.get('track_memory_usage', False):
    report['memory-usage'] = torch.cuda.memory_stats(device)
    break  # Останавливает обучение после первой эпохи!
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: использовать для production обучения
config['track_memory_usage'] = True
# Обучение остановится после первой эпохи!
```

### Когда использовать

Только для профилирования:
```python
config['track_memory_usage'] = True
# Запустить одну эпоху
# Посмотреть report['memory-usage']
# Установить False и запустить снова
```

---

## 11. eval_parts должна содержать 'val'

### Проблема

```python
# bin/tabpack/tabpack.py:1483-1484
eval_parts = config.get('eval_parts', ['val', 'test'])
assert 'val' in eval_parts
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО:
config['eval_parts'] = ['test']  # Assert упадёт!
```

### Почему это важно

- `val` используется для early stopping (`state.update()` сравнивает val metrics)
- `val` используется для online ensemble updates (`update_part='val'`)
- Без `val` невозможно определить улучшение модели

---

## 12. amp_dtype поддерживает только bfloat16

### Проблема

```python
# bin/tabpack/tabpack.py:955-960
def _make_autocast(amp_dtype, device):
    dtype = lib.util.get_amp_dtype(amp_dtype, device)
    assert dtype is torch.bfloat16, 'For now, only "bfloat16" is supported as amp_dtype'
    return torch.autocast(device.type, dtype)
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО:
config['amp_dtype'] = 'float16'  # Assert упадёт!
```

### Почему float16 не поддерживается

- Float16 требует gradient scaling для стабильности
- Gradient scaling не реализован для pack-оптимизаторов
- Bfloat16 имеет больший диапазон и не требует scaling

### Рекомендация

- Современные GPU (A100, H100, RTX 30xx+): `amp_dtype='bfloat16'`
- Старые GPU: `amp_dtype=None` (без AMP)

---

## 13. batch_size влияет на gradient scale

### Проблема

Размер батча влияет на масштаб градиентов:

```python
# bin/tabpack/tabpack.py:1588-1594
losses = loss_fn(apply_model(...), Y_train[batch_idx])
loss = losses.sum()  # Сумма по pack, среднее по batch (внутри loss_fn)
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: изменить batch_size без корректировки lr
config['batch_size'] = 256
config['optimizer']['lr'] = 0.001

# Увеличить batch_size в 4 раза:
config['batch_size'] = 1024
config['optimizer']['lr'] = 0.001  # Learning rate слишком маленький!
```

### Рекомендация

При изменении `batch_size` пропорционально менять `lr`:
```python
# Linear scaling rule:
new_lr = old_lr * (new_batch_size / old_batch_size)
```

---

## 14. module_pack_load_state_dict требует правильные индексы

### Проблема

```python
# bin/tabpack/nn.py:884-899
def module_pack_load_state_dict(module, state_dict, *, pack_idx, state_dict_idx=None):
    for name, x in module.named_parameters():
        if isinstance(x, ParameterPack):
            x.data[pack_idx] = state_dict.pop(name)[
                pack_idx if state_dict_idx is None else state_dict_idx
            ]
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: загрузить веса без pack_idx
model.load_state_dict(state_dict)  # Не работает для pack!

# НЕПРАВИЛЬНО: перепутать pack_idx и state_dict_idx
module_pack_load_state_dict(
    model, state_dict,
    pack_idx=torch.tensor([0, 1, 2]),  # Куда загружать
    state_dict_idx=torch.tensor([5, 6, 7]),  # Откуда брать
)
# Если state_dict имеет pack_size=3, то индексы [5, 6, 7] выйдут за границы!
```

### Правильное использование

```python
# Загрузить веса моделей [0, 1, 2] из state_dict в позиции [3, 4, 5] pack
module_pack_load_state_dict(
    model, state_dict,
    pack_idx=torch.tensor([3, 4, 5]),  # Куда загружать
    state_dict_idx=torch.tensor([0, 1, 2]),  # Откуда брать
)
```

### Где это критично

- При остановке моделей — загрузка лучших весов
- При инференсе — загрузка весов ансамбля
- При resume — загрузка чекпоинта

---

## 15. MLPBackbonePack создаёт max_n_blocks блоков

### Проблема

```python
# bin/tabpack/nn.py:493-505
self.blocks = nn.ModuleList([
    cls.Block(
        in_features=d_in if i == 0 else d_block,
        out_features=d_block,
        ...
    )
    for i in range(max_n_blocks)  # Создаются ВСЕ блоки!
])
```

### Что это значит

Если `n_blocks = [1, 2, 3]` и `max_n_blocks = 3`:
- Создаются 3 блока
- Модель 0 использует только блок 0
- Модель 1 использует блоки 0-1
- Модель 2 использует блоки 0-2

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: предполагать что блоки удаляются
model.backbone.blocks  # Всегда имеет длину max_n_blocks!

# НЕПРАВИЛЬНО: итерировать по всем блокам для всех моделей
for block in model.backbone.blocks:
    x = block(x)  # Модели с меньшим n_blocks получат лишние преобразования!
```

### Как это работает

```python
# bin/tabpack/nn.py:524-534
def _compute_block_idx_list(self):
    # Возвращает список: для каждого блока — какие модели активны
    # Блок 0: [0, 1, 2]  # Все модели
    # Блок 1: [1, 2]     # Модели с n_blocks >= 2
    # Блок 2: [2]        # Модели с n_blocks >= 3
```

### Последствия

- Память выделяется под все блоки
- Веса неактивных блоков не обновляются (градиенты = 0)
- При инференсе нужно использовать `_compute_block_idx_list()`

---

## 16. OneHotEncoding и OOV категории

### Проблема

```python
# bin/tabpack/nn.py:25-57
class OneHotEncoding(nn.Module):
    def forward(self, x):
        return torch.cat([
            nn.functional.one_hot(x[..., i], cardinality + 1)[..., :-1]
            for i, cardinality in enumerate(self._cardinalities)
        ], -1)
```

### Что это значит

- In-vocabulary категории: one-hot encoding
- Out-of-vocabulary категории (индекс `cardinality`): all-zeros encoding

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: предполагать что OOV обрабатывается specially
# OOV получает all-zeros, что может быть не оптимально

# НЕПРАВИЛЬНО: использовать cardinality без +1
nn.functional.one_hot(x, cardinality)  # OOV категории вызовут ошибку!
```

### Где это критично

- При инференсе на новых данных с неизвестными категориями
- При добавлении новых категориальных признаков

---

## 17. DataPreprocessor: fit перед transform

### Проблема

```python
# lib/data.py:41-97
class DataPreprocessor:
    def __init__(self, config):
        self._fitted = False
    
    def fit(self, dataset):
        # Обучает трансформеры на train
        self._fitted = True
        return self
    
    def transform(self, dataset):
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        # Применяет трансформеры
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: вызвать transform без fit
preprocessor = DataPreprocessor(config)
dataset = preprocessor.transform(dataset)  # RuntimeError!

# НЕПРАВИЛЬНО: fit на всех данных (data leakage)
preprocessor.fit(dataset)  # dataset содержит train, val, test
# Трансформеры обучатся на val и test!
```

### Правильное использование

```python
# Fit только на train, transform на всех
preprocessor = DataPreprocessor(config)
dataset = preprocessor.fit_transform(dataset)  # Fit на train, transform на всех
```

### При инференсе

```python
# Загрузить сохранённый preprocessor
preprocessor = artifact['preprocessor']
# Применить transform БЕЗ fit
new_dataset = preprocessor.transform(new_dataset)
```

---

## 18. Legacy vs новый формат state_dicts

### Проблема

```python
# Legacy формат:
state_dicts = {
    0: {...},  # model_id
    1: {...},
}

# Новый формат:
state_dicts = {
    (0, 123060): {...},  # (model_id, step)
    (1, 228540): {...},
}
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: предполагать один формат
for model_id, state_dict in state_dicts.items():
    # Если format новый, model_id будет tuple!
    model.load_state_dict(state_dict)  # Неправильно!
```

### Правильная обработка

```python
# infer.py:
first_key = next(iter(state_dicts.keys()))
is_new_format = isinstance(first_key, tuple)

if is_new_format:
    for (model_id, step), state_dict in state_dicts.items():
        # Обработка нового формата
else:
    for model_id, state_dict in state_dicts.items():
        # Обработка legacy формата
```

---

## 19. generate_training_batches: разные батчи для каждой модели

### Проблема

```python
# bin/tabpack/tabpack.py:851-866
def generate_training_batches(*, train_size, batch_size, batch_generator, pack_size):
    random_values = torch.rand(
        (pack_size, train_size),  # Разные случайные значения для каждой модели!
        generator=batch_generator,
        device=batch_generator.device,
    )
    batches = random_values.argsort(dim=BATCH_DIM).split(batch_size, dim=BATCH_DIM)
    return list(batches)
```

### Что это значит

- Каждая модель получает свой порядок батчей
- Это увеличивает разнообразие обучения
- Батчи имеют форму `(pack_size, batch_size)`

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: предполагать одинаковые батчи для всех моделей
for batch_idx in batches:
    # batch_idx имеет форму (pack_size, batch_size)
    x = dataset.x_train[batch_idx]  # Неправильно!
    
# ПРАВИЛЬНО:
for batch_idx in batches:
    for i in range(pack_size):
        x_i = dataset.x_train[batch_idx[i]]  # Батч для модели i
```

---

## 20. StatePack.update: сравнение по val score

### Проблема

```python
# bin/tabpack/tabpack.py:384-413
def update(self, metrics, *, predictions, predictions_torch, model_state_dict):
    if self.best_metrics:
        improved_mask = metrics['val']['score'] > self.best_metrics['val']['score']
        improved_pack_idx = np.nonzero(improved_mask)[0]
        
        if len(improved_pack_idx) > 0:
            self.n_consequtive_bad_updates[improved_pack_idx] = 0
            # Обновление best_metrics, best_predictions, best_model_state_dicts
```

### Что это значит

- Улучшение определяется **только по val score**
- Test metrics сохраняются, но не влияют на early stopping
- Train metrics сохраняются для анализа

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: использовать test score для early stopping
# Это data leakage!

# НЕПРАВИЛЬНО: предполагать что все metrics одинаково важны
# Только val['score'] определяет улучшение
```

---

## 21. Checkpoint store удаляет старые чекпоинты

### Проблема

```python
# bin/tabpack/tabpack.py:1843-1851
# Remove old ensemble checkpoints that are no longer in the current ensemble
keys_to_remove = [
    key for key in checkpoint_store.get_all_checkpoints().keys()
    if key not in current_ensemble_keys
]

for key in keys_to_remove:
    checkpoint_store.remove_checkpoint(*key)
```

### Что это значит

- При каждом улучшении ансамбля старые чекпоинты удаляются
- Это предотвращает утечку памяти
- Но теряется история ансамблей

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: предполагать что все чекпоинты сохраняются
# Только чекпоинты текущего ансамбля сохраняются!

# НЕПРАВИЛЬНО: отключить удаление (утечка памяти)
# if False:  # Не делайте так!
#     for key in keys_to_remove:
#         checkpoint_store.remove_checkpoint(*key)
```

---

## 22. apply_model_impl: извлечение данных из Dataset

### Проблема

```python
# bin/tabpack/tabpack.py:158-171
def apply_model_impl(model, dataset, *, part, batch_idx=None):
    x_num = dataset.data.get('x_num', {}).get(part)
    x_cat = dataset.data.get('x_cat', {}).get(part)
    
    if batch_idx is not None:
        if x_num is not None:
            x_num = x_num[batch_idx]
        if x_cat is not None:
            x_cat = x_cat[batch_idx]
    
    return model(x_num, x_cat)
```

### Что это значит

- `apply_model_impl` извлекает данные из `dataset.data`
- Поддерживает как numpy, так и torch tensors
- `batch_idx` может быть None (все данные) или Tensor (батч)

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: передать данные напрямую
model(x_num, x_cat)  # Пропускает batch_idx логику!

# НЕПРАВИЛЬНО: предполагать что batch_idx — это список индексов
batch_idx = [0, 1, 2, 3]  # Должен быть Tensor!
```

---

## 23. _make_optimizer: pack_size параметр

### Проблема

```python
# bin/tabpack/tabpack.py:1034-1043
def _make_optimizer(type, **kwargs):
    optimizer_cls = getattr(torch.optim, type, None)
    if optimizer_cls is None:
        optimizer_cls = {
            x.__name__: x
            for x in [AdamWPack, MuonAdamWPack]
        }[type]
    if 'pack_size' not in inspect.signature(optimizer_cls.__init__).parameters:
        kwargs.pop('pack_size', None)  # Удаляется для стандартных оптимизаторов!
    return optimizer_cls(**kwargs)
```

### Что это значит

- Pack-оптимизаторы требуют `pack_size`
- Стандартные оптимизаторы (AdamW, SGD) не поддерживают `pack_size`
- `pack_size` автоматически удаляется для стандартных оптимизаторов

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: использовать стандартный optimizer с per-model lr
config['optimizer'] = {
    'type': 'AdamW',  # Стандартный optimizer
    'lr': [0.001, 0.002, 0.003],  # Per-model lr не поддерживается!
}

# ПРАВИЛЬНО:
config['optimizer'] = {
    'type': 'AdamWPack',  # Pack optimizer
    'lr': [0.001, 0.002, 0.003],  # Per-model lr поддерживается
}
```

---

## 24. OnlineEnsemble: update_type влияет на pool

### Проблема

```python
# bin/tabpack/tabpack.py:678-721
def _prepare_pool(self, ...):
    if self._update_type == 'final':
        pool_ids = finished_ids
        pool_predictions = finished_predictions
    else:
        if self._update_type == 'best':
            running_predictions = running_best_predictions
        else:  # 'latest'
            running_predictions = running_latest_predictions
        
        pool_ids = np.concat([finished_ids, running_ids])
        pool_predictions = {
            k: np.concat([finished_predictions[k], running_predictions[k]])
            for k in running_predictions.keys()
        }
```

### Что это значит

- `'final'` — только завершённые модели
- `'best'` — завершённые + лучшие чекпоинты работающих
- `'latest'` — завершённые + текущие предсказания работающих

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: использовать 'final' в начале обучения
config['online_ensembles'] = {
    'greedy': {
        'type': 'greedy',
        'update_type': 'final',  # Pool пуст пока нет завершённых моделей!
    }
}
# Ансамбль не будет обновляться до первой остановленной модели!
```

### Рекомендация

- Начало обучения: `'latest'` или `'best'`
- Стабильное обучение: `'best'`
- Быстрое прототипирование: `'latest'`

---

## 25. HyperparameterSampler: _BASIC_SAMPLERS

### Проблема

```python
# bin/tabpack/tabpack.py:178-183
class HyperparameterSampler:
    _BASIC_SAMPLERS = (
        'BruteForceSampler',
        'GridSampler',
        'RandomSampler',
        'QMCSampler',
    )
```

### Что это значит

```python
# bin/tabpack/tabpack.py:1179-1187
def _validate_config(config):
    sampler_config = config.get('sampler')
    if (
        sampler_config is not None
        and sampler_config.get('type') not in HyperparameterSampler._BASIC_SAMPLERS
    ):
        raise ValueError(
            'Given the provided sampler config, pack_size must be set explicitly,'
            ' because it can have non-trivial impact on the results'
        )
```

### Что НЕЛЬЗЯ делать

```python
# НЕПРАВИЛЬНО: использовать TPESampler без явного n_models
config['sampler'] = {
    'type': 'TPESampler',  # Не в _BASIC_SAMPLERS!
}
config['n_models'] = 64  # Должно быть задано явно!

# Если n_models не задано — ValueError!
```

### Почему это важно

- TPESampler адаптируется к результатам
- Количество trials может влиять на результаты
- `n_models` должно быть задано явно для воспроизводимости

---

## Чеклист перед внесением изменений

### При добавлении нового loss:

- [ ] Loss возвращает тензор формы `(pack_size,)` после `mean(BATCH_DIM)`
- [ ] `loss.sum()` в training loop даёт правильные градиенты
- [ ] Метрики в `metrics.py` и `metrics_torch.py` обновлены
- [ ] `prediction_type` совместим с новым loss

### При добавлении нового optimizer:

- [ ] Optimizer поддерживает `pack_size` параметр
- [ ] Per-model гиперпараметры передаются как списки
- [ ] `optimizer_pack_remove()` работает корректно
- [ ] `configs_T` flow обновлён если нужны per-model параметры

### При изменении модели:

- [ ] `forward()` возвращает правильную форму `(pack_size, batch_size, output_dim)`
- [ ] `PACK_DIM=0`, `BATCH_DIM=1` соблюдаются
- [ ] `module_pack_load_state_dict()` работает с новой структурой
- [ ] `_prepare_model_config()` обрабатывает новые per-model параметры
- [ ] `max_*` размерности выводятся правильно

### При изменении ансамбля:

- [ ] Алгоритм возвращает `(ensemble_idx, ensemble_weights)`
- [ ] `OnlineEnsemble` обновлён если нужны новые параметры
- [ ] Ensemble snapshot сохраняется правильно
- [ ] `update_type` совместим с новым алгоритмом

### При изменении данных:

- [ ] `DataPreprocessor.fit()` обучает только на train
- [ ] `DataPreprocessor.transform()` применяется ко всем частям
- [ ] `regression_label_stats` сохраняется и используется
- [ ] `prediction_type` совместим с новыми данными

---

## Ссылки

- [`_make_loss_fn_pack()`](../bin/tabpack/tabpack.py:1046)
- [`pack_remove()`](../bin/tabpack/tabpack.py:505)
- [`_evaluate()`](../bin/tabpack/tabpack.py:879)
- [`OnlineEnsemble._prepare_pool()`](../bin/tabpack/tabpack.py:678)
- [`module_pack_load_state_dict()`](../bin/tabpack/nn.py:884)
- [`MLPBackbonePack._compute_block_idx_list()`](../bin/tabpack/nn.py:524)
- [`DataPreprocessor`](../lib/data.py:41)

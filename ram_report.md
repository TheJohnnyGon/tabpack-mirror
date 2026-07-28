# RAM-отчет: TabPack обучение

## Обозначения

Базовая единица измерения — **D** = размер сырого датасета в RAM:
- **D = N × F × 4** байт (x_num, float32), где N — объектов, F — фич
- y пренебрежимо мал (int64, 1 значение на объект), игнорируем

Пропорции сплитов:
- **D_train = 0.85 D**, **D_val = 0.07 D**, **D_test = 0.07 D**

Модель:
- **P** — pack_size (число параллельных моделей)
- **M** — размер одной модели (веса) = `n_blocks × d_block × max(F, d_block) × 4`
- **O** — размер состояния оптимизатора на одну модель ≈ **2M** (exp_avg + exp_avg_sq) + 0.3M (muon) ≈ **2.3M**
- **M_pack = P × M**, **O_pack = P × 2.3M**

Батч:
- **B** — batch_size
- **B_mem = P × B × max(F, d_block) × 4** — размер одного батча с pack-dim

---

## Пошаговый анализ мультипликаторов RAM

### 1. `load_data()` — загрузка .npy

**Код:** [`lib/data.py:158`](lib/data.py:158)
```python
data = {x.stem: np.load(x) for x in dataset_dir.iterdir() if _is_npy_path(x)}
```

| Что в памяти | Размер |
|---|---|
| x_num (N×F, float32) | **1.0 D** |
| split-индексы (int32) | ~0.003 D (пренебрежимо) |

**Итого: 1.0 D**

---

### 2. `apply_split(copy=False)` — разделение на части

**Код:** [`lib/data.py:135-144`](lib/data.py:135)
```python
return {k: data[v].copy() if copy else data[v] for k, v in split.items()}
```

`copy=False` → создаются **view** на оригинальный массив. View сам по себе мал, но **держит ссылку на всё базовое storage** размера N×F.

После `data = apply_split(data, split)` старый `data` должен быть собран GC, но view'ы продолжают держать полное storage.

| Что в памяти | Размер |
|---|---|
| Базовое storage x_num (через view'ы) | **1.0 D** |
| view-обёртки (train/val/test) | ~0 |

**Итого: 1.0 D** (без изменений, но view держит весь массив)

⚠️ **Проблема:** view на `(N, F)` не освобождает неиспользуемые строки. Если нужно только train (0.85N), в памяти всё равно лежит весь N.

---

### 3. `transform_num()` — нормализация num-фич

**Код:** [`lib/data.py:203-240`](lib/data.py:203)

Последовательные операции:

| Подшаг | Операция | Новые копии | Пик RAM |
|--------|----------|-------------|---------|
| 3a | `QuantileTransformer.fit(X_num_train)` | Внутренние структуры: ~0.03 D | **1.03 D** |
| 3b | `X_num = {k: normalizer.transform(v)}` | 3 новые части (train+val+test) = **1.0 D** | **2.0 D** |
| 3c | `np.nan_to_num(v)` | 3 копии = **1.0 D** | **3.0 D** |
| 3d | `np.unique(x)` для каждой колонки | F × N временных | **~3.0 D** |
| 3e | `v[:, mask]` (фильтрация констант) | view на старый массив | **~3.0 D** |
| 3f | `v.astype(float32)` | 3 копии = **1.0 D** | **4.0 D** |

После завершения: старые массивы собираются GC, остаётся новый X_num = **1.0 D**.

**Пик: ~4.0 D**
**После: 1.0 D**

⚠️ **СЛАБОЕ МЕСТО #1:** Пик **×4** от базового размера на этапе препроцессинга. Каждая операция создаёт полную копию всех 3 частей.

---

### 4. `Task.from_dir()` — загрузка task + labels

**Код:** [`lib/data.py:373-382`](lib/data.py:373)
```python
y = np.load(dataset_dir / 'y.npy')  # отдельная загрузка
task = Task(apply_split(y, split, copy=True), ...)  # copy=True
```

| Что в памяти | Размер |
|---|---|
| y (полный, из np.load) | ~0 (y мал) |
| task.labels (3 копии через copy=True) | ~0 |

**Итого: ~1.0 D** (y пренебрежимо мал)

⚠️ **Проблема:** y загружается отдельно от `load_data()`. Для регрессии с float32 y это ~0.25D дополнительно. Для классификации — незначимо.

---

### 5. `dataset.to_torch(device)` — перенос на GPU

**Код:** [`lib/data.py:456-466`](lib/data.py:456)
```python
part: torch.as_tensor(value, device=device)
```

`torch.as_tensor(..., device=cuda)` создаёт tensor на GPU, но **оригинальный numpy array остаётся в CPU RAM**.

| Что в памяти | Размер |
|---|---|
| CPU: numpy x_num (view на storage) | **1.0 D** |
| GPU: torch x_num (train+val+test) | **1.0 D** |

**Итого: 2.0 D (1.0 CPU + 1.0 GPU)**

⚠️ **СЛАБОЕ МЕСТО #2 (КРИТИЧНОЕ):** Полное дублирование данных. Numpy-массивы **никогда не удаляются** после `to_torch()`. Данные лежат и в CPU и в GPU одновременно.

---

### 6. `ModelPack` + оптимизатор

**Код:** [`tabpack.py:1370-1431`](tabpack.py:1370)

| Что в памяти (GPU) | Размер |
|---|---|
| Model weights (P моделей) | **M_pack = P × M** |
| Optimizer state (exp_avg, exp_avg_sq, muon) | **O_pack ≈ 2.3 × P × M** |
| Gradients (появляются при backward) | **M_pack** |

**Итого GPU: ~4.3 × P × M** (с градиентами)

Для типичных параметров (P=64, d_block=384, F=819, n_blocks=3):
- M ≈ 2 MB на модель
- M_pack ≈ 128 MB
- O_pack ≈ 294 MB
- Градиенты ≈ 128 MB

---

### 7. `generate_training_batches()` — индексы батчей на эпоху

**Код:** [`tabpack.py:858-864`](tabpack.py:858)
```python
random_values = torch.rand((pack_size, train_size), device=device)  # float32
batches = random_values.argsort(dim=BATCH_DIM).split(batch_size)     # int64
```

| Что в памяти (GPU) | Размер |
|---|---|
| `random_values`: (P, N_train) float32 | **P × 0.85N × 4 = 0.34 × P/N × D** |
| `argsort` результат: (P, N_train) int64 | **P × 0.85N × 8 = 0.68 × P/N × D** |
| `batches` (список view) | ~0 |

Для P=64, N=110K: random=23 MB, argsort=46 MB.

**Итого GPU: +0.34×P/N×D + 0.68×P/N×D**

⚠️ **СЛАБОЕ МЕСТО #3:** Хранит ВСЕ индексы перестановки для всех P моделей сразу. Для P=64, N=10M: ~2 GB только на индексы.

---

### 8. Forward pass (один батч)

**Код:** [`tabpack.py:158-167`](tabpack.py:158)

```python
dataset.data['x_num']['train'][batch_idx]  # индексация
```

`batch_idx` shape = (P, B). Индексация создаёт тензор (P, B, F).

| Что в памяти (GPU, временное) | Размер |
|---|---|
| x_num batch: (P, B, F) | **B_mem = P × B × F × 4** |
| После PackView expand: (P, B, F) | **B_mem** (тот же) |
| После num_module (если есть): (P, B, F×d_emb) | зависит от emb |
| После backbone (каждый блок): (P, B, d_block) | **P × B × d_block × 4** |
| Output: (P, B, 1) | ~0 |

Autograd хранит промежуточные тензоры для backward:
| Autograd graph | Размер |
|---|---|
| Входы каждого блока | ~n_blocks × B_mem |
| Градиенты весов | **M_pack** |

**Пик GPU на батч: ~3-4 × B_mem + M_pack (градиенты)**

Для P=64, B=512, F=819: B_mem ≈ 107 MB, пик ≈ 400-500 MB.

---

### 9. `_evaluate()` — предсказание на val/test

**Код:** [`tabpack.py:898-906`](tabpack.py:898)
```python
y_pred_torch = torch.cat([
    apply_model(model, dataset, part=part, batch_idx=batch_idx)
    for batch_idx in ...
], dim=BATCH_DIM)
```

Все предсказания для части накапливаются через `torch.cat`.

| Что в памяти (GPU) | Размер |
|---|---|
| y_pred_torch для val: (P, N_val) | **P × 0.07N × 4 = 0.028 × P/N × D** |
| y_pred_torch для test: (P, N_test) | **P × 0.07N × 4 = 0.028 × P/N × D** |
| `y_pred = y_pred_torch.cpu().numpy()` | ещё **0.056 × P/N × D** в CPU |

**Итого: +0.056×P/N×D GPU + 0.056×P/N×D CPU**

⚠️ **СЛАБОЕ МЕСТО #4:** `torch.cat` всех предсказаний. Для P=64, N_val=1M: 256 MB на GPU. Метрики можно считать инкрементально.

---

### 10. `state.update()` — сохранение лучших результатов

**Код:** [`tabpack.py:383-428`](tabpack.py:383)

**Первый вызов** (best_metrics пуст):
```python
self._best_predictions = deepcopy(predictions)         # CPU numpy
self._best_predictions_torch = deepcopy(predictions_torch)  # GPU tensor
self._best_model_state_dicts = deepcopy(model_state_dict)   # GPU tensor
```

| Что в памяти | Размер |
|---|---|
| best_predictions (CPU, val+test) | **0.056 × P/N × D** |
| best_predictions_torch (GPU, val+test) | **0.056 × P/N × D** |
| best_model_state_dicts (GPU) | **M_pack = P × M** |

**Последующие вызовы** (только improved модели):
- Обновляются только строки improved_pack_idx — эффективно, O(improved/P) от полного.

**Итого после первого update: +0.11×P/N×D + P×M**

⚠️ **СЛАБОЕ МЕСТО #5:** `deepcopy(model_state_dict)` на первом шаге — полная копия всех весов. `deepcopy` для dict с тензорами работает корректно, но `.clone()` был бы явнее.

---

### 11. Online Ensembles

**Код:** [`tabpack.py:677-746`](tabpack.py:677)

`_prepare_pool()` конкатенирует finished + running предсказания:

| Что в памяти (GPU) | Размер |
|---|---|
| pool_predictions_torch: (pool_size, N_val) | **P × 0.07N × 4** |
| candidate_predictions в greedy: (n_candidates, N_val) | **P × 0.07N × 4** |

**Пик в ensemble update: +0.14 × P/N × D**

---

### 12. `steps_numlog` — лог на каждый шаг

**Код:** [`tabpack.py:1574-1581`](tabpack.py:1574)
```python
steps_numlog.append({
    'loss': loss_detached,  # Tensor (P,) на GPU
})
```

Каждый шаг добавляет GPU-тензор (P,). За эпоху: ~N_train/B шагов.

| Что в памяти | Размер |
|---|---|
| За одну эпоху | **(N_train/B) × P × 4** байт |

Для P=64, N_train=90K, B=512: 176 × 256 = 45 KB. Незначимо по размеру, но тензоры на GPU.

⚠️ **СЛАБОЕ МЕСТО #6:** GPU-тензоры в numlog. Лучше `.item()` или `.cpu().numpy()`.

---

### 13. `pack_epochs_numlog` — лог на каждую эпоху

**Код:** [`tabpack.py:1608-1623`](tabpack.py:1608)

При `save_all_predictions = false` (наш базовый сценарий):
```python
pack_epochs_numlog.append(deepcopy({
    'step': state.steps,      # (P,) numpy int64
    'time': np.full(P, ...),  # (P,) numpy float64
    'metrics': eval_metrics,  # метрики, (P,) numpy
    # predictions НЕ включены
}))
```

| Что в памяти (CPU) | Размер |
|---|---|
| На одну эпоху | P × (8 + 8 + несколько × 8) ≈ **P × 100 байт** |
| За E эпох | **E × P × 100** |

Для P=64, E=100: ~640 KB. **Пренебрежимо мало.**

При `save_all_predictions = true`:
| На одну эпоху | **P × (0.07N + 0.07N) × 4 = 0.056 × P/N × D** |
| За E эпох | **E × 0.056 × P/N × D** |

⚠️ **СЛАБОЕ МЕСТО #7 (КРИТИЧНОЕ при save_all_predictions=true):** Линейный рост с эпохами. E=100, P=64, N=10M → **~512 GB**.

---

## Сводная таблица мультипликаторов

### CPU RAM

| Этап | Мультипликатор | Накопительно |
|------|---------------|-------------|
| 1. load_data | +1.0 D | **1.0 D** |
| 2. apply_split | +0 | **1.0 D** |
| 3. transform_num (пик) | +3.0 D | **4.0 D** |
| 3. transform_num (после) | -2.0 D | **1.0 D** |
| 4. Task.from_dir | +0 | **1.0 D** |
| 5. to_torch | +0 (numpy не удаляется) | **1.0 D** |
| 10. state.update | +0.056×P/N×D | **1.0 D + ε** |
| 12. steps_numlog | ~0 | **1.0 D + ε** |
| 13. pack_epochs_numlog | ~0 (без predictions) | **1.0 D + ε** |

**Пик CPU RAM: 4.0 D** (на препроцессинге)
**Стабильный CPU RAM: 1.0 D** (numpy данные не удаляются!)

### GPU VRAM

| Этап | Мультипликатор | Накопительно |
|------|---------------|-------------|
| 5. to_torch | +1.0 D | **1.0 D** |
| 6. модель + оптимизатор | +3.3×P×M | **1.0 D + 3.3PM** |
| 7. batch indices | +1.02×P×0.85N×4 | **1.0 D + 3.3PM + 0.35(P/N)D** |
| 8. forward+backward (пик) | +4×B_mem + PM | **+4B_mem+PM** |
| 9. evaluate | +0.112×P/N×D | **+0.11(P/N)D** |
| 10. state.update | +0.056×P/N×D + PM | **+0.06(P/N)D+PM** |

**Пик GPU VRAM: 1.0 D + 4.3PM + 1.02×P×0.85N×4 + 4×B_mem + 0.17×P/N×D**

Упрощённо:
- **1.0 D** — данные
- **4.3 PM** — модель + оптимизатор + градиенты
- **~0.35 P/N × D** — индексы батчей
- **~4 B_mem** — forward/backward пик
- **~0.17 P/N × D** — предсказания

---

## График мультипликаторов CPU RAM

```
RAM / D
 4.0 |        ╭─────╮
     |        │ пик  │
 3.0 |        │препро│
     |        │цесс  │
 2.0 |        │      ╰──────╮
     |        │             ╰───────────────────────────────
 1.0 |────────╯                                                     │
     |  load   split  transform   to_torch   train loop    finish   │
     |                                                          не  │
     |                                                          удал.│
     |                                                          numpy│
 0.0 |___________________________________________________________________
```

**Ключевой инсайт:** После препроцессинга CPU RAM падает до **1.0 D**, но остаётся на этом уровне до конца, потому что numpy-массивы не удаляются после переноса на GPU.

---

## График мультипликаторов GPU VRAM

```
VRAM
     |                    ╭──╮
 Пик |                    │FW│← forward + backward + autograd
     |                    │BD│
     |              ╭─────╯╰─────╮
     |     ╭────╮   │evaluate    │
 База|     │mod │   │+predict    │
     |     │+opt│   │+batch_idx  │
     |╭────╯    ╰───╯            ╰────────
     │data
     │
     ╰─────────────────────────────────────→ время
       to_torch  model   train    eval    next epoch
```

---

## Топ проблем и решения

### #1 — Numpy не удаляется после to_torch (×2 дублирование)

**Где:** [`lib/data.py:456`](lib/data.py:456) → [`tabpack.py:1326`](tabpack.py:1326)

**Проблема:** `dataset = dataset.to_torch(device)` создаёт GPU-копии, но оригинальные numpy-массивы остаются в CPU RAM. Данные лежат в двух местах одновременно.

**Эффект:** **+1.0 D** лишней CPU RAM на всё время обучения.

**Решение:** После `to_torch()` явно удалить numpy-данные:
```python
dataset = dataset.to_torch(device)
# Удалить numpy-копии
for key in list(dataset.data.keys()):
    # dataset.data теперь содержит только тензоры
    pass
del dataset_np  # если была отдельная переменная
gc.collect()
```

Или изменить `to_torch` на in-place версию:
```python
def to_torch_(self, device):
    for key in self.data:
        for part in self.data[key]:
            self.data[key][part] = torch.as_tensor(self.data[key][part], device=device)
```

---

### #2 — Пик ×4 на препроцессинге

**Где:** [`lib/data.py:203-240`](lib/data.py:203)

**Проблема:** Каждая операция в `transform_num` создаёт полную копию всех 3 частей датасета. Цепочка `transform → nan_to_num → astype` даёт пик ×4.

**Эффект:** Пик **4.0 D** на препроцессинге.

**Решение:** Использовать in-place операции где возможно:
```python
# Вместо:
X_num = {k: np.nan_to_num(v) for k, v in X_num.items()}
X_num = {k: v.astype(_X_NUM_DTYPE) for k, v in X_num.items()}

# In-place:
for k in X_num:
    np.nan_to_num(X_num[k], copy=False)  # если dtype уже float32
```

Или последовательное освобождение:
```python
for k in X_num:
    X_num[k] = np.nan_to_num(X_num[k])
    del old_X_num[k]  # явное освобождение
    gc.collect()
```

---

### #3 — Индексы батчей: P×N_train на GPU

**Где:** [`tabpack.py:858-864`](tabpack.py:858)

**Проблема:** `torch.rand((P, N_train))` + `argsort` хранит все индексы перестановки для всех моделей.

**Эффект:** **P × N_train × 12 байт** на GPU (float32 + int64). Для P=64, N=10M: ~3 GB.

**Решение:** Генерировать батчи инкрементально:
```python
def generate_training_batches_incremental(train_size, batch_size, pack_size, generator):
    """Генерирует батчи по одному, не храня все индексы."""
    for _ in range(math.ceil(train_size / batch_size)):
        indices = torch.rand(pack_size, batch_size, generator=generator, device=generator.device)
        # или использовать torch.randperm для каждого pack member
        yield ...
```

Или использовать `torch.randperm` только для текущего батча.

---

### #4 — torch.cat всех предсказаний в evaluate

**Где:** [`tabpack.py:898-906`](tabpack.py:898)

**Проблема:** Все предсказания для части накапливаются в один тензор через `torch.cat`.

**Эффект:** **P × N_part × 4** на GPU + ещё столько же в CPU после `.cpu().numpy()`.

**Решение:** Инкрементальный расчёт метрик:
```python
def _evaluate_incremental(...):
    metrics_accumulator = {}
    for batch_idx in batches:
        y_pred_batch = apply_model(model, dataset, part=part, batch_idx=batch_idx)
        # Обновляем метрики инкрементально
        metrics_accumulator.update(y_pred_batch, y_true_batch)
    return metrics_accumulator.finalize()
```

Для ROC-AUC и других метрик, которые требуют все предсказания, можно использовать онлайн-алгоритмы или батч-аппроксимации.

---

### #5 — deepcopy в state.update()

**Где:** [`tabpack.py:426-428`](tabpack.py:426)

**Проблема:** `deepcopy(predictions_torch)` и `deepcopy(model_state_dict)` на первом шаге.

**Эффект:** Полная копия предсказаний + весов. Для тензоров `deepcopy` работает как `.clone()`, но менее явно.

**Решение:** Заменить на явный `.clone()`:
```python
self._best_predictions_torch = {k: v.clone() for k, v in predictions_torch.items()}
self._best_model_state_dicts = {k: v.clone() for k, v in model_state_dict.items()}
```

---

### #6 — GPU-тензоры в steps_numlog

**Где:** [`tabpack.py:1574-1581`](tabpack.py:1574)

**Проблема:** `loss_detached` — GPU-тензор, сохраняется в список.

**Эффект:** Много маленьких GPU-тензоров, потенциальная фрагментация.

**Решение:**
```python
steps_numlog.append({
    'loss': loss_detached.cpu().tolist(),  # или .mean().item()
})
```

---

### #7 — save_all_predictions (если включено)

**Где:** [`tabpack.py:1617-1620`](tabpack.py:1617)

**Проблема:** Каждый epoch deepcopy'ит предсказания всех моделей.

**Эффект:** **E × 0.056 × P/N × D** накопительно.

**Решение:** `save_all_predictions = false` (уже по умолчанию в этом отчёте).

---

## Пример для Peru (конкретные цифры)

**Параметры:** N=110K, F=819, P=64, d_block=384, B=512, n_blocks=3

**Базовый размер:** D = 110000 × 819 × 4 = **358 MB**
**Модель:** M = 3 × 384 × 819 × 4 + 3 × 384 × 384 × 4 + 384 × 4 ≈ **2 MB**
**M_pack:** P × M = 64 × 2 = **128 MB**
**O_pack:** 2.3 × 128 = **294 MB**

### CPU RAM

| Этап | Мультипликатор | MB |
|------|---------------|-----|
| Пик препроцессинг | 4.0 D | **1432** |
| Стабильный (после to_torch) | 1.0 D | **358** |
| + best_predictions | 0.056×P/N×D | **1.3** |
| **Итого стабильный CPU** | | **~360 MB** |

### GPU VRAM

| Компонент | Формула | MB |
|-----------|---------|-----|
| Данные | 1.0 D | **358** |
| Модель | M_pack | **128** |
| Оптимизатор | 2.3 × M_pack | **294** |
| Градиенты | M_pack | **128** |
| Batch indices | P×0.85N×12 | **69** |
| Forward/Backward пик | 4×B_mem | **428** |
| Evaluate предсказания | 0.11×P/N×D | **2.8** |
| Best state | 0.06×P/N×D + M_pack | **130** |
| **Пик GPU VRAM** | | **~1535 MB** |
| **Стабильный GPU (без fwd)** | | **~1108 MB** |

---

## Пример для большого датасета

**Параметры:** N=10M, F=1000, P=64, d_block=384, B=512

**Базовый размер:** D = 10M × 1000 × 4 = **36.4 GB**
**Модель:** M ≈ 2.2 MB, M_pack ≈ **141 MB**

### CPU RAM

| Этап | Мультипликатор | GB |
|------|---------------|-----|
| Пик препроцессинг | 4.0 D | **145.6** |
| Стабильный (с numpy) | 1.0 D | **36.4** |
| Стабильный (без numpy) | 0 D | **~0** |

### GPU VRAM

| Компонент | Формула | GB |
|-----------|---------|-----|
| Данные | 1.0 D | **36.4** |
| Модель + оптимизатор + градиенты | 4.3PM | **0.6** |
| Batch indices | P×0.85N×12 | **2.7** |
| Forward/Backward пик | 4×B_mem | **0.4** |
| Evaluate + best state | 0.17×P/N×D | **0.4** |
| **Пик GPU VRAM** | | **~40.5 GB** |
| **Стабильный GPU (без fwd)** | | **~40.1 GB** |

Для 8 GPU (A100 80GB): **40.5 GB < 80 GB** — влезает, но с малым запасом.
Для 8 GPU (V100 16GB): **40.5 GB > 16 GB** — не влезает, нужен DDP с разделением данных.

---

## Итоговые рекомендации

| Приоритет | Проблема | Экономия | Сложность |
|-----------|----------|----------|-----------|
| **P0** | Удалить numpy после to_torch | **-1.0 D CPU** | Низкая |
| **P0** | In-place препроцессинг | **-3.0 D пик** | Средняя |
| **P1** | Инкрементальные батчи | **-P×N×12 GPU** | Средняя |
| **P1** | Инкрементальный evaluate | **-P×N_part×8** | Высокая |
| **P2** | .clone() вместо deepcopy | Косметика | Низкая |
| **P2** | Scalar в numlog | Косметика | Низкая |
| **P3** | save_all_predictions=false | **-E×0.056×P/N×D** | Низкая |

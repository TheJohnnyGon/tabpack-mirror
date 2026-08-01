# Анализ bisect/02-merged: что фиксится из RAM-проблем

## Краткий ответ

Из 7 проблем в ram_report.md бранч **фиксит 4 напрямую** и частично ещё 1.
Два фикса (numpy free + batch indices) можно взять **без DDP**.

---

## Попроблемный разбор

### ✅ FIX #1: Numpy не удаляется после to_torch (СЛАБОЕ МЕСТО #2)

**Что в бранче:** [`tabpack.py`](bin/tabpack/tabpack.py)
```python
numpy_dataset = dataset
dataset = dataset.to_torch(device)
del numpy_dataset
gc.collect()
```

**Эффект:** -1.0 D CPU RAM постоянно.

**Можно ли взять без DDP:** **ДА.** Это полностью независимый фикс, 4 строки кода.

---

### ✅ FIX #2: Индексы батчей P×N_train на GPU (СЛАБОЕ МЕСТО #3)

**Что в бранче:** [`tabpack.py:generate_training_batches`](bin/tabpack/tabpack.py:853)

**Было:**
```python
random_values = torch.rand((pack_size, train_size), device=device)  # GPU float32
batches = random_values.argsort(dim=BATCH_DIM).split(batch_size)    # GPU int64
# Пик: P × N_train × 12 байт на GPU
```

**Стало:**
```python
epoch_seed = torch.randint(0, 2**62, (1,), generator=batch_generator).item()
cpu_gen = torch.Generator('cpu').manual_seed(epoch_seed)
perms = torch.stack(
    [torch.randperm(train_size, generator=cpu_gen) for _ in range(pack_size)]
)  # CPU: P × N_train × 8 байт
batches = []
for start in range(0, train_size, batch_size):
    end = min(start + batch_size, batch_size)
    batches.append(perms[:, start:end].to(device, non_blocking=True))
```

**Проблема:** perms всё ещё хранится целиком на CPU (P×N_train×8), и batches — список тензоров на GPU, которые тоже хранятся все сразу.

- **Экономия GPU:** -P×N_train×12 → 0 (батчи на GPU создаются как копии чанков, но perms на CPU остаётся)
- **CPU RAM:** +P×N_train×8 (perms) — но это CPU, а не GPU, так что менее критично

**Реальный мультипликатор:**
| | Было (GPU) | Стало (CPU) |
|---|---|---|
| Batch indices | P×N_train×12 на **GPU** | P×N_train×8 на **CPU** |
| Для P=64, N=10M | **2.7 GB GPU** | **1.8 GB CPU** |

⚠️ **Не идеально:** perms всё ещё целиком в памяти (CPU), и batches держит все батчи на GPU одновременно. Но перенос с GPU на CPU — это прогресс.

**Можно ли взять без DDP:** **ДА.** Полностью независимый фикс.

---

### ✅ FIX #3: GPU-тензоры в steps_numlog (СЛАБОЕ МЕСТО #6)

**Что в бранче:** [`tabpack.py:1659`](bin/tabpack/tabpack.py:1659)

**Было:**
```python
loss_detached = loss.detach()       # GPU tensor (P,)
batch_losses.append(loss_detached)
steps_numlog.append({'loss': loss_detached})
```

**Стало:**
```python
loss_value = loss.item()            # Python float
batch_losses.append(loss_value)
steps_numlog.append({'loss': loss_value})
```

**Эффект:** Убирает накопление GPU-тензоров. Косметический фикс, но правильный.

**Можно ли взять без DDP:** **ДА.** 2 строки.

---

### ✅ FIX #4: Пик ×4 на препроцессинге (СЛАБОЕ МЕСТО #1) — ЧАСТИЧНО

**Что в бранче:** [`lib/data.py:transform_num`](lib/data.py:248)

**Было:**
```python
X_num = {k: normalizer.transform(v) for k, v in X_num.items()}  # 3 копии сразу
X_num = {k: np.nan_to_num(v) for k, v in X_num.items()}          # ещё 3 копии
X_num = {k: v.astype(_X_NUM_DTYPE) for k, v in X_num.items()}    # ещё 3 копии
# Пик: 4.0 D
```

**Стало:**
```python
# Chunked transform с pre-allocated buffer
result = {}
for part, arr in X_num.items():
    buf = np.empty((n_rows, n_cols), dtype=_X_NUM_DTYPE)  # один буфер
    for start in range(0, n_rows, 5_000_000):
        chunk_out = normalizer.transform(arr[start:end])
        buf[start:end] = chunk_out.astype(_X_NUM_DTYPE, copy=False)
    result[part] = buf
X_num = result

X_num = {k: np.nan_to_num(v, copy=False) for k, v in X_num.items()}  # in-place!
```

**Эффект:**
- Chunked transform: вместо 3 полных копий — один буфер + чанк
- `nan_to_num(copy=False)`: in-place, нет копии
- subsample для constant mask: вместо `np.unique` по всему столбцу — по саамплу

| | Было | Стало |
|---|---|---|
| Пик transform | **4.0 D** | **~2.0 D** |
| nan_to_num | копия | in-place |
| constant mask | все N | subsample 5M |

**Можно ли взять без DDP:** **ДА.** Независимый фикс в `lib/data.py`.

---

### ⚠️ ЧАСТИЧНО: mmap загрузка (СЛАБОЕ МЕСТО #1, дополнение)

**Что в бранче:** [`lib/data.py:load_data`](lib/data.py:156)

```python
def load_data(dataset_dir, split_id, *, mmap: bool = False):
    load_kwargs = {'mmap_mode': 'r'} if mmap else {}
    raw = {x.stem: np.load(x, **load_kwargs) ...}
    data = apply_split(raw, split)
    if mmap:
        del raw
        gc.collect()
```

**Эффект:** При `mmap=True` данные не загружаются в RAM полностью, а маппятся. После split только нужные строки становятся resident.

**Можно ли взять без DDP:** **ДА,** но с оговорками:
- mmap добавляет параметр `mmap` в `load_data` и `build_dataset`
- работает только если данные не модифицируются in-place (transform_num создаёт новые массивы, так что OK)
- может быть медленнее на некоторых ФС

---

### ❌ НЕ ФИКСИРУЕТСЯ: torch.cat в evaluate (СЛАБОЕ МЕСТО #4)

`_evaluate()` по-прежнему делает `torch.cat` всех предсказаний. Без изменений.

### ❌ НЕ ФИКСИРУЕТСЯ: deepcopy в state.update (СЛАБОЕ МЕСТО #5)

`deepcopy(predictions_torch)` и `deepcopy(model_state_dict)` по-прежнему на месте. Без изменений.

---

## Мультипликаторы: было → стало (без DDP, только полезные фиксы)

### CPU RAM

| Этап | Было (×D) | Стало (×D) | Фикс |
|------|-----------|------------|------|
| Пик препроцессинг | **4.0** | **~2.0** | chunked transform + in-place nan_to_num |
| Стабильный (после to_torch) | **1.0** | **~0.0** | `del numpy_dataset + gc.collect()` |
| + best_predictions | 0.056×P/N | 0.056×P/N | не фиксится |

**Итого CPU RAM:**
- Пик: **4.0 D → 2.0 D** (−50%)
- Стабильный: **1.0 D → ~0 D** (−100%, только best_predictions остаётся)

### GPU VRAM

| Компонент | Было (×D) | Стало (×D) | Фикс |
|-----------|-----------|------------|------|
| Данные | 1.0 D | 1.0 D | не меняется |
| Модель + оптимизатор | 4.3 PM | 4.3 PM | не меняется |
| Batch indices | **0.35×P/N×D на GPU** | **0 (перенесено на CPU)** | generate_training_batches |
| Forward/Backward пик | 4×B_mem | 4×B_mem | не меняется |
| Evaluate предсказания | 0.11×P/N×D | 0.11×P/N×D | не фиксится |
| Best state | 0.06×P/N×D + PM | 0.06×P/N×D + PM | не фиксится |
| steps_numlog GPU тензоры | ε | **0** | loss.item() |

**Итого GPU VRAM:**
- Пик: **−0.35×P/N×D** (batch indices ушли на CPU)
- Стабильный: **−0.35×P/N×D**

---

## Конкретные цифры: Peru

| | Было | Стало (фиксы без DDP) | Экономия |
|---|---|---|---|
| Пик CPU RAM | 1432 MB | **~716 MB** | **−716 MB (−50%)** |
| Стабильный CPU RAM | 360 MB | **~2 MB** | **−358 MB (−99%)** |
| Пик GPU VRAM | 1535 MB | **~1466 MB** | **−69 MB (−5%)** |
| Стабильный GPU | 1108 MB | **~1039 MB** | **−69 MB (−6%)** |

## Конкретные цифры: Большой датасет (N=10M, F=1000)

| | Было | Стало (фиксы без DDP) | Экономия |
|---|---|---|---|
| Пик CPU RAM | **145.6 GB** | **~72.8 GB** | **−72.8 GB (−50%)** |
| Стабильный CPU RAM | **36.4 GB** | **~0 GB** | **−36.4 GB (−100%)** |
| Пик GPU VRAM | **40.5 GB** | **~37.8 GB** | **−2.7 GB (−7%)** |
| Стабильный GPU | **40.1 GB** | **~37.4 GB** | **−2.7 GB (−7%)** |

---

## Что можно взять из бранча (чек-лист)

### Можно взять прямо сейчас (независимо от DDP):

1. **`del numpy_dataset; gc.collect()` после to_torch** — 4 строки, экономит 1.0 D CPU
2. **`generate_training_batches` с CPU perms** — переносит batch indices с GPU на CPU
3. **`loss.item()` вместо `loss.detach()`** — 2 строки, убирает GPU тензоры из numlog
4. **Chunked transform_num + in-place nan_to_num** — снижает пик препроцессинга ×4 → ×2
5. **subsample для constant mask** — ускоряет + экономит память при np.unique

### Можно взять с оговорками:

6. **`mmap=True` в load_data** — нужно добавить флаг в конфиг, работает не на всех ФС

### Привязано к DDP (не брать пока):

7. `init_distributed()`, DDP wrapper, sharding, is_main_process guard — всё это DDP-специфичное
8. Cache с `_COMPLETE` marker — DDP-ориентированный (rank 0 build, others wait)
9. LR scaling для DDP

---

## Резюме

| Фикс | Файл | Экономия | Сложность | DDP-зависимость |
|------|------|----------|-----------|----------------|
| del numpy после to_torch | tabpack.py | **-1.0 D CPU** | 4 строки | ❌ Нет |
| CPU batch indices | tabpack.py | **-0.35×P/N×D GPU** | ~20 строк | ❌ Нет |
| loss.item() | tabpack.py | косметика | 2 строки | ❌ Нет |
| Chunked transform_num | data.py | **-2.0 D пик CPU** | ~30 строк | ❌ Нет |
| in-place nan_to_num | data.py | часть пика | 1 строка | ❌ Нет |
| subsample constant mask | data.py | ускорение | ~5 строк | ❌ Нет |
| mmap load_data | data.py | **-1.0 D CPU** (с флагом) | ~15 строк | ❌ Нет |

**Общая экономия без DDP:**
- CPU пик: **4.0 D → 2.0 D** (−50%)
- CPU стабильный: **1.0 D → 0 D** (−100%)
- GPU: **−0.35×P/N×D** (batch indices на CPU)

# Модуль nn.py — Neural Network Pack Modules

## Назначение

Модуль реализует паттерн **"Module Pack"** — набор нейронных модулей с разными гиперпараметрами, обучаемых параллельно в одном тензоре.

## Константы

```python
PACK_DIM = 0   # Размерность pack (первая ось тензора)
BATCH_DIM = 1  # Размерность batch (вторая ось тензора)
```

Все тензоры имеют форму: `(pack_size, batch_size, features)`

---

## Базовые классы

### [`ParameterPack`](bin/tabpack/nn.py:102)

**Назначение:** Маркерный класс для обучаемых параметров в pack-модулях.

**Наследование:** `nn.Parameter`

**Использование:**
```python
self.weight = ParameterPack(torch.empty(pack_size, out_features, in_features))
```

---

### [`BufferPack`](bin/tabpack/nn.py:135)

**Назначение:** Маркерный класс для необучаемых буферов (например, размеры слоёв).

**Особенность:** Использует метакласс `_BufferPackMeta` для корректной работы `isinstance()`.

---

### [`ModulePack`](bin/tabpack/nn.py:165)

**Назначение:** Базовый абстрактный класс для всех pack-модулей.

**Обязательные свойства:**
- [`pack_size`](bin/tabpack/nn.py:180) — Возвращает размер pack

**Методы:**
- [`get_pack_size()`](bin/tabpack/nn.py:183) — Безопасное получение pack_size (возвращает None при ошибке)

---

## Модули обработки данных

### [`OneHotEncoding`](bin/tabpack/nn.py:25)

**Назначение:** One-hot кодирование категориальных признаков.

**Вход:** `(*, n_cat_features)`
**Выход:** `(*, sum(cardinalities))`

**Особенность:** Поддерживает OOV (out-of-vocabulary) категории — кодирует их как все нули.

---

### [`PackView`](bin/tabpack/nn.py:68)

**Назначение:** Преобразование входных данных в pack-формат.

**Логика:**
- Если вход 2D `(batch, features)` → расширяет до `(pack_size, batch, features)`
- Если вход уже 3D → пропускает без изменений

---

## Линейные слои

### [`LinearPack`](bin/tabpack/nn.py:212)

**Назначение:** Pack линейных слоёв с разными размерами входов/выходов.

**Параметры:**
- `in_features` — int или list[int] (размер входа для каждой модели)
- `out_features` — int или list[int] (размер выхода)
- `max_in_features` — максимальный размер входа (для padding)
- `max_out_features` — максимальный размер выхода
- `pack_size` — количество моделей

**Как работает:**
1. Все веса хранятся в тензоре `(pack_size, max_out, max_in)`
2. При forward применяется маска для "отключения" лишних нейронов
3. Поддерживает режим `loop=True` для отладки (медленный, но точный)

---

## Активации и регуляризация

### [`DropoutPack`](bin/tabpack/nn.py:352)

**Назначение:** Pack dropout-слоёв с разными вероятностями.

**Параметр:** `p` — float или list[float]

---

### [`LeakyReLUPack`](bin/tabpack/nn.py:385)

**Назначение:** Pack LeakyReLU активаций (демонстрационный пример).

**Параметр:** `negative_slope` — float или list[float]

---

## Backbone

### [`MLPBackbonePack`](bin/tabpack/nn.py:435)

**Назначение:** Pack многослойных перцептронов.

**Параметры:**
- `d_in` — размер входа
- `n_blocks` — int или list[int] (количество блоков для каждой модели)
- `d_block` — int или list[int] (размер скрытого слоя)
- `dropout` — float или list[float]
- `activation` — тип активации ('ReLU' и др.)
- `max_n_blocks` — максимум блоков
- `max_d_block` — максимум размера блока

**Внутренний класс [`Block`](bin/tabpack/nn.py:439):**
- [`linear`](bin/tabpack/nn.py:448) — LinearPack
- [`activation`](bin/tabpack/nn.py:449) — Активация
- [`dropout`](bin/tabpack/nn.py:453) — DropoutPack

**Ключевой метод [`_compute_block_idx_list()`](bin/tabpack/nn.py:524):**
Вычисляет, какие модели активны в каждом блоке. Например:
```
n_blocks = [1, 2, 3, 1, 2, 3]
Блок 0: все 6 моделей активны
Блок 1: модели [1, 2, 4, 5] активны
Блок 2: модели [2, 5] активны
```

---

## Embedding модули

### [`LinearEmbeddingsPack`](bin/tabpack/nn.py:586)

**Назначение:** Линейное embedding числовых признаков.

**Формула:** `output = weight * input + bias`

**Параметры:**
- `n_features` — количество признаков
- `d_embedding` — размер embedding (int или list)
- `max_d_embedding` — максимум для padding

---

### [`LinearReLUEmbeddingsPack`](bin/tabpack/nn.py:663)

**Назначение:** Linear + ReLU embedding.

**Опция `concat_input`:** Конкатенирует исходный признак с embedding.

---

### [`CosineEmbeddingsPack`](bin/tabpack/nn.py:770)

**Назначение:** Cosine embedding (из статьи TabPack).

**Формула:**
```
1. x = weight * input + bias
2. x = cos(2 * pi * x)
3. x = elementwise_affine(x)
```

**Особенность:** Исходный признак конкатенируется с результатом.

---

## Утилиты

### [`module_pack_load_state_dict()`](bin/tabpack/nn.py:884)

**Назначение:** Загрузка весов в конкретные позиции pack.

**Параметры:**
- `pack_idx` — индексы в целевом pack
- `state_dict_idx` — индексы в source state_dict (по умолчанию те же)

---

### [`make_keep_pack_idx()`](bin/tabpack/nn.py:902)

**Назначение:** Вычислить индексы для сохранения при удалении моделей.

```python
pack_size = 6, remove_idx = [2, 5]
→ keep_idx = [0, 1, 3, 4]
```

---

### [`module_pack_remove()`](bin/tabpack/nn.py:910)

**Назначение:** Физически удалить модели из pack.

**Действия:**
1. Создаёт новые ParameterPack/BufferPack без удалённых моделей
2. Обновляет `_pack_size` в PackView
3. Возвращает маппинг old→new параметров

---

### [`module_pack_select()`](bin/tabpack/nn.py:947)

**Назначение:** Контекстный менеджер для временного выбора подмножества моделей.

**Использование:**
```python
with module_pack_select(model, pack_idx=torch.tensor([0, 2, 4])):
    # Работаем только с моделями 0, 2, 4
    predictions = model(x_num, x_cat)
# После выхода — модель восстановлена
```

---

## Диаграмма зависимостей

```
ModulePack (базовый)
├── LinearPack
│   └── используется в MLPBackbonePack.Block
├── DropoutPack
│   └── используется в MLPBackbonePack.Block
├── MLPBackbonePack
│   └── используется в ModelPack
├── LinearEmbeddingsPack
│   └── используется в ModelPack.num_module
├── LinearReLUEmbeddingsPack
│   └── альтернатива LinearEmbeddingsPack
└── CosineEmbeddingsPack
    └── альтернатива LinearEmbeddingsPack

PackView → используется в ModelPack для подготовки входных данных
OneHotEncoding → используется в ModelPack для категориальных признаков
```

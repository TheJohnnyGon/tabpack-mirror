# TabPack Architecture Documentation

## Обзор проекта

TabPack — это **единая модель-ансамбль** для табличных данных, обучаемая как пакет параллельных нейронных сетей.

### Аналогия с CatBoost

Так же как **CatBoost** — это не одно дерево, а ансамбль из множества деревьев, **TabPack** — это не одна MLP-сеть, а ансамбль из множества MLP-сетей с разными гиперпараметрами. Результат работы TabPack — это не предсказание одной модели, а усреднённое (или взвешенное) предсказание всего ансамбля.

Каждая отдельная MLP в ансамбле обучается с разными гиперпараметрами (разная глубина, ширина, dropout, learning rate), но все они работают над одной задачей и в финале объединяются в единую модель.

### Технические детали

Под капотом TabPack использует концепцию **"Pack"** — пакетной обработки моделей с разными гиперпараметрами в одном тензоре, что позволяет обучать весь ансамбль параллельно на GPU за один forward/backward проход.

### Ключевые концепции

1. **Pack (Пакет)** — набор моделей одинаковой архитектуры, но с разными гиперпараметрами, обучаемых одновременно
2. **Parallel Training** — все модели в pack обучаются параллельно через batch operations
3. **Online Ensembling** — динамическое формирование ансамбля во время обучения
4. **Early Stopping per Model** — каждая модель в pack может остановиться независимо

---

## Структура проекта

```
bin/tabpack/          # Основной код обучения
├── tabpack.py        # Главная функция обучения (main)
├── nn.py             # Neural Network модули (ModulePack паттерн)
├── optim.py          # Оптимизаторы (AdamWPack, MuonAdamWPack)
├── metrics.py        # Метрики для pack-тензоров
├── ensemble_utils_torch.py  # Алгоритмы ансамблирования
├── ensemble_checkpoint.py   # Хранение чекпоинтов ансамбля
└── utils.py          # Вспомогательные функции

lib/
├── data.py           # Загрузка и преобразование данных
├── metrics.py        # Базовые метрики
└── types.py          # Базовые типы данных

infer/
└── infer.py          # Инференс обученных моделей
```

---

## Основные классы и их зависимости

### 1. [`ModelPack`](bin/tabpack/tabpack.py:58) — Основная модель

**Назначение:** Инкапсулирует одну нейронную сеть с поддержкой pack-архитектуры.

**Компоненты:**
- [`num_module`](bin/tabpack/tabpack.py:72) — Embedding для числовых признаков (LinearEmbeddingsPack, CosineEmbeddingsPack и др.)
- [`cat_module`](bin/tabpack/tabpack.py:82) — OneHotEncoding для категориальных признаков
- [`pack_view`](bin/tabpack/tabpack.py:87) — Преобразование входных данных в pack-формат
- [`backbone`](bin/tabpack/tabpack.py:88) — MLPBackbonePack (основная нейросеть)
- [`output`](bin/tabpack/tabpack.py:93) — LinearPack (выходной слой)

**Связи:**
- Использует классы из [`nn.py`](bin/tabpack/nn.py:1)
- Работает с [`StatePack`](bin/tabpack/tabpack.py:260) для отслеживания состояния

---

### 2. [`StatePack`](bin/tabpack/tabpack.py:260) — Состояние обучения pack

**Назначение:** Хранит динамическое состояние всех моделей в pack.

**Ключевые свойства:**
- [`ids`](bin/tabpack/tabpack.py:280) — ID каждой модели в pack
- [`configs`](bin/tabpack/tabpack.py:284) — Конфигурации гиперпараметров
- [`steps`](bin/tabpack/tabpack.py:288) — Текущий шаг обучения каждой модели
- [`n_consequtive_bad_updates`](bin/tabpack/tabpack.py:292) — Счётчик плохих обновлений (для early stopping)
- [`best_metrics`](bin/tabpack/tabpack.py:296) — Лучшие метрики для каждой модели
- [`best_predictions`](bin/tabpack/tabpack.py:304) — Лучшие предсказания
- [`best_model_state_dicts`](bin/tabpack/tabpack.py:312) — Лучшие веса моделей

**Методы:**
- [`step()`](bin/tabpack/tabpack.py:340) — Инкрементирует счётчик шагов
- [`remove()`](bin/tabpack/tabpack.py:343) — Удаляет завершённые модели из pack
- [`update()`](bin/tabpack/tabpack.py:384) — Обновляет best_metrics при улучшении

---

### 3. [`OnlineEnsemble`](bin/tabpack/tabpack.py:603) — Онлайн ансамбль

**Назначение:** Динамически формирует ансамбль моделей во время обучения.

**Параметры:**
- [`type`](bin/tabpack/tabpack.py:608) — Тип алгоритма (greedy, autotopk, bruteforce, beam)
- [`update_type`](bin/tabpack/tabpack.py:612) — 'final', 'best' или 'latest' предсказания
- [`patience`](bin/tabpack/tabpack.py:617) — Терпение перед остановкой обновления

**Методы:**
- [`update()`](bin/tabpack/tabpack.py:749) — Обновляет ансамбль новыми моделями
- [`_prepare_pool()`](bin/tabpack/tabpack.py:678) — Формирует пул кандидатов

---

### 4. [`HyperparameterSampler`](bin/tabpack/tabpack.py:174) — Сэмплер гиперпараметров

**Назначение:** Обёртка над Optuna для генерации конфигураций моделей.

**Методы:**
- [`ask()`](bin/tabpack/tabpack.py:218) — Запрашивает новую конфигурацию
- [`tell()`](bin/tabpack/tabpack.py:228) — Сообщает результат обучения

---

## Функции обучения

### [`main()`](bin/tabpack/tabpack.py:1299) — Главная функция

**Полный цикл обучения:**

1. **Инициализация**
   - Загрузка данных через [`lib.data.Dataset.from_dir()`](lib/data.py:931)
   - Препроцессинг через [`DataPreprocessor`](lib/data.py:54)
   - Создание [`ModelPack`](bin/tabpack/tabpack.py:58)

2. **Training Loop**
   - Генерация батчей через [`generate_training_batches()`](bin/tabpack/tabpack.py:851)
   - Forward pass через [`apply_model_impl()`](bin/tabpack/tabpack.py:158)
   - Loss computation через [`_make_loss_fn_pack()`](bin/tabpack/tabpack.py:1046)
   - Backward pass и optimizer step

3. **Evaluation**
   - Вызов [`_evaluate()`](bin/tabpack/tabpack.py:879) для всех частей (val, test)
   - Обновление [`StatePack`](bin/tabpack/tabpack.py:260)
   - Проверка early stopping через [`compute_stop_pack_idx()`](bin/tabpack/tabpack.py:519)

4. **Ensembling**
   - Обновление [`OnlineEnsemble`](bin/tabpack/tabpack.py:603)
   - Сохранение чекпоинтов через [`EnsembleCheckpointStore`](bin/tabpack/ensemble_checkpoint.py:19)

5. **Сохранение**
   - Dump [`model.pt`](bin/tabpack/tabpack.py:2076) с весами и конфигом
   - Dump [`report.json`](bin/tabpack/tabpack.py:1974) с метриками

---

## Neural Network модули

### [`ModulePack`](bin/tabpack/nn.py:165) — Базовый класс

Базовый класс для всех pack-модулей. Обеспечивает:
- Поддержку pack dimension (PACK_DIM = 0)
- Хранение параметров через [`ParameterPack`](bin/tabpack/nn.py:102)
- Хранение буферов через [`BufferPack`](bin/tabpack/nn.py:135)

### [`LinearPack`](bin/tabpack/nn.py:212) — Линейный слой

Поддерживает разные размеры входов/выходов для каждой модели в pack.

### [`MLPBackbonePack`](bin/tabpack/nn.py:435) — MLP backbone

Реализует многослойный перцептрон с поддержкой разного количества блоков.

### [`DropoutPack`](bin/tabpack/nn.py:352) — Dropout

Поддерживает разные dropout rates для каждой модели.

---

## Оптимизаторы

### [`AdamWPack`](bin/tabpack/optim.py:99)

Pack-версия AdamW с поддержкой разных learning rates.

### [`MuonAdamWPack`](bin/tabpack/optim.py:218)

Гибридный оптимизатор: Muon для линейных слоёв, AdamW для остальных.

---

## Работа с данными

### [`DataPreprocessor`](lib/data.py:54)

**Паттерн fit/transform:**
- [`fit()`](lib/data.py:98) — Обучает трансформеры на train данных
- [`transform()`](lib/data.py:225) — Применяет трансформеры ко всем частям
- [`fit_transform()`](lib/data.py:366) — Комбинация fit + transform

**Поддерживает:**
- Извлечение бинарных признаков из числовых
- StandardScaler / QuantileTransformer для числовых
- OrdinalEncoder / OneHotEncoder для категориальных

### [`Dataset`](lib/data.py:920)

Dataclass для хранения данных и задачи.

---

## Инференс

### [`infer.py`](infer/infer.py:1)

**Функции:**
- [`load_model_artifact()`](infer/infer.py:78) — Загрузка model.pt
- [`build_model_for_ensemble()`](infer/infer.py:83) — Восстановление модели
- [`evaluate_ensemble()`](infer/infer.py:139) — Оценка ансамбля

---

## Алгоритмы ансамблирования

### В [`ensemble_utils_torch.py`](bin/tabpack/ensemble_utils_torch.py:1)

| Функция | Описание |
|---------|----------|
| [`topk_ensemble()`](bin/tabpack/ensemble_utils_torch.py:97) | Выбирает top-K моделей по score |
| [`autotopk_ensemble()`](bin/tabpack/ensemble_utils_torch.py:110) | Автоматически определяет размер ансамбля |
| [`bruteforce_ensemble()`](bin/tabpack/ensemble_utils_torch.py:141) | Перебор всех комбинаций |
| [`greedy_ensemble()`](bin/tabpack/ensemble_utils_torch.py:160) | Жадный алгоритм с весами |
| [`beam_ensemble()`](bin/tabpack/ensemble_utils_torch.py:339) | Beam search по комбинациям |

---

## Поток данных

```
Данные → DataPreprocessor → Dataset → ModelPack → Optimizer → StatePack
                                                        ↓
                                                  OnlineEnsemble
                                                        ↓
                                                  model.pt
```

---

## Конфигурация

### Основные параметры [`Config`](bin/tabpack/tabpack.py:1133)

| Параметр | Описание |
|----------|----------|
| `n_models` | Количество моделей в pack |
| `model` | Конфигурация модели |
| `optimizer` | Конфигурация оптимизатора |
| `batch_size` | Размер батча |
| `n_epochs` | Максимум эпох |
| `patience` | Patience для early stopping |
| `online_ensembles` | Конфигурация онлайн ансамблей |
| `sampler` | Конфигурация сэмплера гиперпараметров |

---

## Ключевые зависимости между модулями

```
tabpack.py
├── nn.py (ModulePack, LinearPack, MLPBackbonePack...)
├── optim.py (AdamWPack, MuonAdamWPack)
├── metrics.py (calculate_metrics_pack)
├── ensemble_utils_torch.py (greedy_ensemble, autotopk...)
├── ensemble_checkpoint.py (EnsembleCheckpointStore)
└── lib/data.py (Dataset, DataPreprocessor, Task)

infer.py
├── tabpack.py (ModelPack, _evaluate)
├── nn.py (module_pack_load_state_dict)
└── lib/data.py (Dataset, DataPreprocessor)
```

---

## Запуск

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

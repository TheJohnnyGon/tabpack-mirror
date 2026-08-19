# Отчёт по работе над TabPack — Июнь–Август 2026

## Содержание

1. [Общая информация](#1-общая-информация)
2. [Хронология работ](#2-хронология-работ)
   - [2.1 Июнь 2026 — Начало работы и воспроизведение статьи](#21-июнь-2026-начало-работы-и-воспроизведение-статьи)
   - [2.2 Июль 2026 — TabPack, Nirvana, оптимизация памяти](#22-июль-2026-tabpack-nirvana-оптимизация-памяти)
   - [2.3 Август 2026 — Pairwise, LazyDataset, RerankService, замер качества](#23-август-2026-pairwise-lazydataset-rerankservice-замер-качества)
3. [Технический разбор по веткам](#3-технический-разбор-по-веткам)
   - [3.1 Ветка `main` — Базовая инфраструктура](#31-ветка-main-базовая-инфраструктура)
   - [3.2 Ветка `pairwise` — Pairwise обучение и конвертация](#32-ветка-pairwise-pairwise-обучение-и-конвертация)
   - [3.3 Ветка `pairwise-lazy` — LazyDataset и оптимизация загрузки](#33-ветка-pairwise-lazy-lazydataset-и-оптимизация-загрузки)
4. [Итоговые результаты](#4-итоговые-результаты)
5. [Точки роста и дальнейшие шаги](#5-точки-роста-и-дальнейшие-шаги)

---

## 1. Общая информация

**Цель:** Замена архитектуры CatBoost на TabPack (эволюция TabM) для ранжирования на L3-стадии GeoSuggest.

**Трек:** ADR по замене архитектуры модели.

**Ключевые ресурсы:**
- Кубик YRec GPU для обучения DL-моделей
- Датасеты L3-саджеста (Колумбия, Пакистан, Перу)
- Offline-replay и Metrics для замера качества

---

## 2. Хронология работ

### 2.1 Июнь 2026 — Начало работы и воспроизведение статьи

**29 июня** — Запись бадди, постановка задачи.

- Изучена архитектура TabM из arXiv статьи
- Определён план работы:
  1. Воспроизвести результаты из arXiv локально
  2. Воспроизвести результаты в Nirvana (YRec GPU кубик)
  3. Подготовить граф для обучения TabM на датасетах L3-саджеста

**Коммиты в ветке `main`:**

| Коммит | Описание |
|--------|----------|
| [`28bac17`](../code/tabpack-ag:1) | Реализован новый преобразователь YT → NPY |
| [`c629030`](../code/tabpack-ag:1) | Переделан `convert_bce` |
| [`81a3fc2`](../code/tabpack-ag:1) | Добавлен `convert_polars` для ускорения конвертации |
| [`d9e9b7f`](../code/tabpack-ag:1) | Добавлено логирование в convert, убраны старые файлы |
| [`cbd3663`](../code/tabpack-ag:1) | Доработан convert |
| [`aef1370`](../code/tabpack-ag:1) | Добавлены вспомогательные функции для convert |
| [`9cfb8e8`](../code/tabpack-ag:1) | Обычные отчёты |
| [`3ebbc34`](../code/tabpack-ag:1) | Добавлены конфиги для Peru и генератор конфигов |
| [`94856d0`](../code/tabpack-ag:1) | Добавлена функция для инференса модели |
| [`c4e80cf`](../code/tabpack-ag:1) | Добавлена функция создания итогового конфига без eval |
| [`4304da2`](../code/tabpack-ag:1) | Добавлен флаг для сохранения модели |
| [`8bf5d30`](../code/tabpack-ag:1) | Ускорена конвертация |
| [`edbcab9`](../code/tabpack-ag:1) | Исправлен `convert_bce.py` |
| [`2beec20`](../code/tabpack-ag:1) | Переименован `parse_results`, добавлена функция сбора метрик запусков |

### 2.2 Июль 2026 — TabPack, Nirvana, оптимизация памяти

**6 июля** — Успешное воспроизведение результатов статьи.

- Экспериментально обнаружен конфиг для многосидовой оценки
- Результаты воспроизведены на датасетах diamond, black-friday, otto

**Результаты воспроизведения TabM:**

| Датасет | Модель | Статья | Ноутбук |
|---------|--------|--------|---------|
| diamond | TabM | 0.1342 ± 0.0017 | 0.1310 ± 0.0007 |
| diamond | TabM† mini | 0.1320 ± 0.0010 | 0.1315 ± 0.0006 |
| black-friday | TabM | 0.6875 ± 0.0015 | 0.6869 ± 0.0004 |
| black-friday | TabM† mini | 0.6807 ± 0.0013 | 0.6781 ± 0.0004 |
| otto | TabM | 0.8268 ± 0.0014 | 0.8275 ± 0.0014 |
| otto | TabM† mini | 0.8342 ± 0.0014 | 0.8342 ± 0.0012 |

**12 июля** — Тестирование новой архитектуры TabPack.

**Результаты TabPack:**

| Датасет | Модель | Статья | Ноутбук |
|---------|--------|--------|---------|
| diamond | TabPack | 0.1331 ± 0.0003 | 0.1333 ± 0.0005 |
| diamond | TabPack† | 0.1307 ± 0.0007 | 0.1302 ± 0.0005 |
| black-friday | TabPack | 0.6877 ± 0.0003 | 0.6876 ± 0.0003 |
| black-friday | TabPack† | 0.6784 ± 0.0002 | 0.6781 ± 0.0003 |
| otto | TabPack | 0.8238 ± 0.0012 | 0.8247 ± 0.0009 |
| otto | TabPack† | 0.8281 ± 0.0017 | 0.8295 ± 0.0021 |

**19 июля** — Запуск обучения в Nirvana на датасете Перу.

- Реализован перевод датасетов из YT в формат для обучения
- Подбор оптимального hidden_dim на подвыборке 1M объектов (BCE loss)

**Результаты подбора hidden_dim:**

| Модель | Hidden-dim | Результат |
|--------|------------|-----------|
| TabPack | 384 | 0.7965 |
| TabPack | 512 | 0.7964 |
| TabPack | 768 | 0.7953 |
| TabPack | 1024 | 0.7956 |
| TabPack† | 384 | **0.8045** |
| TabPack† | 512 | 0.8038 |
| TabPack† | 768 | 0.8035 |

**27 июля** — Результаты Pointwise обучения на реальных данных.

| Метрика | CatBoost | TabPack | TabPack-cosine |
|---------|----------|---------|----------------|
| ROC-AUC | 0.897724 | 0.906136 | **0.910094** |
| Accuracy | 0.821130 | 0.828628 | **0.832616** |
| F1 | 0.809390 | 0.828230 | **0.832268** |

**Проблемы, выявленные на этом этапе:**
- Большие датасеты не влезают в 1 GPU (10+ минут на эпоху)
- Пиковое потребление RAM до 800 ГБ при датасете 80 ГБ (OOM)
- Кривой препроцессинг, отсутствие шардов и mmap

**Коммиты в ветке `main` (продолжение):**

| Коммит | Описание |
|--------|----------|
| [`5651a06`](../code/tabpack-ag:1) | Оптимизация потребления памяти: gc.collect(), cuda.empty_cache(), del промежуточных массивов |
| [`cc825dd`](../code/tabpack-ag:1) | Логгирование этапов загрузки и предобработки датасета |
| [`4e48920`](../code/tabpack-ag:1) | `skip_bin_encoder` — пропуск OrdinalEncoder когда фичи уже 0/1 |
| [`319e9b3`](../code/tabpack-ag:1) | Fix UnboundLocalError в `_extract_bin_from_num` |
| [`9673357`](../code/tabpack-ag:1) | `num_memory_efficient` — columnwise обработка числовых признаков (O(N×1) вместо O(N×K)) |
| [`2bdda3b`](../code/tabpack-ag:1) | Скрипт инференса (`infer/infer.py`) + сохранение feature_indices в model.pt |
| [`ee3d4ad`](../code/tabpack-ag:1) | Сохранение информации о greedy ансамбле в model.pt |
| [`72f003c`](../code/tabpack-ag:1) | Сохранение всех моделей greedy ансамбля в model.pt |
| [`79997ab`](../code/tabpack-ag:1) | Сохранение только моделей из greedy ансамбля |
| [`833b720`](../code/tabpack-ag:1) | Сохранение текущих весов для greedy ансамбля с update_type=latest |
| [`6f8aa8d`](../code/tabpack-ag:1) | Сохранение текущих весов для работающих моделей в greedy ансамбле |
| [`04a1597`](../code/tabpack-ag:1) | Обновлены конфиги до текущих |
| [`c08ea78`](../code/tabpack-ag:1) | Save ensemble snapshot on improvement для exact reproducibility |
| [`9f87df3`](../code/tabpack-ag:1) | Save and reuse fitted transformers для exact reproducibility |
| [`e61f4e6`](../code/tabpack-ag:1) | Удалены случайно коммиченные тестовые данные |
| [`371c947`](../code/tabpack-ag:1) | Удалена старая инференска, добавлено сохранение шагов модели в model.pt |
| [`16c073d`](../code/tabpack-ag:1) | Обновлено сохранение весов |
| [`2356e25`](../code/tabpack-ag:1) | Fix сохранение весов моделей в model.pt |
| [`7fed3bc`](../code/tabpack-ag:1) | Заменён best на latest |
| [`c50cd65`](../code/tabpack-ag:1) | Fix сохранение весов в model.pt |
| [`9e2ba03`](../code/tabpack-ag:1) | Добавлен seed в конфиг |
| [`2f69fd9`](../code/tabpack-ag:1) | DataPreprocessor с паттерном fit/transform |
| [`abd34d6`](../code/tabpack-ag:1) | Fix порядок операций в DataPreprocessor.fit() |
| [`86cf040`](../code/tabpack-ag:1) | Fix DataPreprocessor для корректной обработки всех параметров |
| [`1718a65`](../code/tabpack-ag:1) | Fix DataPreprocessor для полного соответствия build_dataset |
| [`3c47b7a`](../code/tabpack-ag:1) | Добавлены .copy()/.clone() для предсказаний в StatePack.update() |
| [`8140883`](../code/tabpack-ag:1) | Убраны clone и copy из StatePack.update() |
| [`bf8005a`](../code/tabpack-ag:1) | Эксперимент с build_dataset() |
| [`bb13138`](../code/tabpack-ag:1) | Возвращён DataPreprocessor |
| [`e6c2716`](../code/tabpack-ag:1) | Оптимизация памяти в DataPreprocessor |
| [`53dc82f`](../code/tabpack-ag:1) | Refactor: EnsembleCheckpointStore вместо dict для хранения весов ансамбля |
| [`94fabd0`](../code/tabpack-ag:1) | Удалены ненужные репорты |
| [`976470c`](../code/tabpack-ag:1) | Собрány репорты в одну папку |
| [`a273aac`](../code/tabpack-ag:1) | Добавлена функция для проверки инференса |
| [`ea57b13`](../code/tabpack-ag:1) | Удалён мусорный код (глобальные переменные, лишний препроцессинг) |
| [`a400839`](../code/tabpack-ag:1) | Добавлена документация для агентов и людей |

### 2.3 Август 2026 — Pairwise, LazyDataset, RerankService, замер качества

**28 июля / 3 августа** — Реализация Pairwise обучения.

**Что было сделано:**

1. **Оптимизация потребления VRAM/RAM:**
   - Удалены лишние копии данных при трансформации
   - Добавлены `del` и `gc.collect()` для промежуточных массивов
   - Оптимизирована работа с OrdinalEncoder
   - Добавлена columnwise обработка числовых фич
   - **Итог:** пиковое потребление RAM снизилось на 40-50%

2. **DataPreprocessor:**
   - Новый класс с `fit()`/`transform()` API
   - Поддержка всех флагов из конфига
   - Переиспользование трансформеров между тренировкой и инференсом

3. **Сохранение модели и инференс:**
   - Аккуратное сохранение дескриптора фич (удаление константных, вычисление бинарных)
   - Отдельное сохранение трансформеров
   - Обработка `include_current_ensemble_in_pool` и `update_type`
   - Модель корректно сохраняется, результаты из Nirvana воспроизводятся локально

4. **Pairwise обучение:**
   - Векторизованное формирование пар по ключу
   - Функция формирования батчей из пар
   - Новый тип обучения в train loop
   - Интеграция с `update_online_ensembles()`
   - Модификация `_evaluate()` для оценки pair accuracy
   - Новый тип задачи в info.json
   - Лоссы `bce_pairwise` и `margin_ranknet`
   - Расширяемый реестр лоссов через `TaskType.PAIRWISE`
   - Переписаны сохранение модели и инференс под pairwise

**Коммиты в ветке `pairwise`:**

| Коммит | Описание |
|--------|----------|
| [`dc0fece`](../code/tabpack-ag:1) | Первая реализация pairwise |
| [`bebff00`](../code/tabpack-ag:1) | Исправлены ошибки |
| [`99eddcf`](../code/tabpack-ag:1) | Fix KeyError: 'train' в tabpack.py |
| [`424d20b`](../code/tabpack-ag:1) | Добавлена поддержка онлайн-ансамблей |
| [`33deb15`](../code/tabpack-ag:1) | Исправлены логи, report и summary |
| [`6e9f5dd`](../code/tabpack-ag:1) | Fix TypeError: cuda tensor → numpy в update_online_ensembles() |
| [`2da9f42`](../code/tabpack-ag:1) | Изменено логгирование |
| [`f2b6571`](../code/tabpack-ag:1) | Refactor: переход на TaskType.PAIRWISE и расширяемый реестр лоссов |
| [`ccc71ce`](../code/tabpack-ag:1) | Убрана рамка для читаемости |
| [`aa97227`](../code/tabpack-ag:1) | Обновлён convert_polars.py: streaming mode для 50M+ строк |
| [`3c7cda8`](../code/tabpack-ag:1) | Fix pickle при параллельной обработке файлов |
| [`2d51f8b`](../code/tabpack-ag:1) | Fix flush() — numpy массивы vs memmap |
| [`7f2d2be`](../code/tabpack-ag:1) | Fix UnboundLocalError: 'eval_pairs_t' |
| [`0915d7b`](../code/tabpack-ag:1) | Refactor convert_polars.py для датасетов 80-100 ГБ |
| [`77d2ac6`](../code/tabpack-ag:1) | Fix обработка батчей в load_tsv() |
| [`172c84d`](../code/tabpack-ag:1) | Выгрузка RawFormulaVal с оригинальными ключами для джойна с серпами |
| [`ae05e90`](../code/tabpack-ag:1) | Fix infer: вывод размеров только существующих партов |
| [`65f5ce1`](../code/tabpack-ag:1) | Добавлен флаг --data-path для переопределения пути к данным |

**10 августа** — Оптимизация convert и LazyDataset.

**Оптимизация convert_polars.py:**
- Чтение через `read_csv_batched` (батчи по 500k строк)
- `ignored-features` убираются в SQL запросе (`generate_yql_query.py`)
- Убраны лишние `.astype()` копии
- Сквозной механизм оригинальных ключей (`original_keys.tsv`)
- Флаг `--metrics` в infer.py для выгрузки RawFormulaVal
- Флаг `--data-path` для переопределения пути к данным
- **Итог:** обработка 80 ГБ занимает ~1 час (было 8 часов)

**LazyDataset:**
- Поддержка обучения с данными на RAM (`data_on_cpu` в конфиге)
- Pinned memory + `non_blocking=True` для CPU→GPU transfer
- `PinnedBatchStager`: преаллоцированные pinned CPU-буферы
- `PrefetchBatchStager`: background thread для overlap GPU compute

**Коммиты в ветке `pairwise-lazy`:**

| Коммит | Описание |
|--------|----------|
| [`4b76d45`](../code/tabpack-ag:1) | DataLoader-подобная загрузка данных для TabPack |
| [`b694a18`](../code/tabpack-ag:1) | Оптимизация lazy-датасета: async H2D + мелкие данные на GPU |
| [`b882b05`](../code/tabpack-ag:1) | Pair indices на CPU при data_on_cpu=True |
| [`d4be6a8`](../code/tabpack-ag:1) | Prefetch pipeline для data_on_cpu=True (Variant 2) |

**Проблема LazyDataset:** обучение с `data_on_cpu=True` замедляется ~13 раз (0:13 → 3:10). Главная причина — `torch.set_num_threads(1)`, из-за чего `torch.index_select` работает в 1 поток на CPU. Prefetch дал ~10% ускорения (3:10 → 2:52).

**17 августа** — RerankService и замер качества.

- Обновлён граф обучения в Nirvana
- Реализован RerankService для Offline-replay
- Инференс CatBoost и TabPack на Непальской корзине
- Замер качества выполнен

**Итоговые метрики (train: 2026-07-05 — 2026-07-29, test: 2026-07-30 — 2026-08-02):**

Модели показывают сопоставимое качество.

---

## 3. Технический разбор по веткам

### 3.1 Ветка `main` — Базовая инфраструктура

**Что было:** Исследовательский код без поддержки сохранения моделей, инференса и работы с большими датасетами.

**Что стало:**

| Компонент | Было | Стало |
|-----------|------|-------|
| **Конвертация данных** | pandas, 8 часов на 80 ГБ | polars + streaming, ~1 час на 80 ГБ |
| **Предобработка** | Встроена в `build_dataset()`, без переиспользования | `DataPreprocessor` с fit/transform API |
| **Сохранение модели** | Не работало для greedy ансамблей | `EnsembleCheckpointStore` с ключом (model_id, step) |
| **Инференс** | Отсутствовал | `infer/infer.py` с поддержкой greedy ансамбля и feature_indices |
| **Память** | Пик 800 ГБ RAM на 80 ГБ датасета | Снижение на 40-50% (columnwise, gc.collect, skip_bin_encoder) |
| **Трансформеры** | Refit при инференсе | Сохраняются в model.pt для exact reproducibility |
| **Документация** | Отсутствовала | 12 MD-файлов в docs/ |

**Ключевые изменения:**

1. **EnsembleCheckpointStore** (`bin/tabpack/ensemble_checkpoint.py`) — новый класс для хранения чекпоинтов с составным ключом (model_id, step). Решает проблему дубликатов моделей в greedy-ансамбле.

2. **DataPreprocessor** — новый класс с паттерном fit/transform, поддерживающий все флаги из конфига. Позволяет переиспользовать трансформеры между тренировкой и инференсом.

3. **Memory optimizations:**
   - `num_memory_efficient()` — columnwise обработка (O(N×1) вместо O(N×K))
   - `skip_bin_encoder` — пропуск OrdinalEncoder для уже бинарных фич
   - `gc.collect()` + `cuda.empty_cache()` после тяжёлых операций
   - `np.ptp` вместо `np.unique` для проверки константных колонок

4. **Inference system** (`infer/infer.py`):
   - Поддержка greedy ансамбля с взвешенным усреднением
   - Сохранение feature_indices (num/cat/bin) для воспроизводимой предобработки
   - Флаг `--metrics` для выгрузки RawFormulaVal с оригинальными ключами
   - Флаг `--data-path` для переопределения пути к данным

### 3.2 Ветка `pairwise` — Pairwise обучение и конвертация

**Что было:** Только Pointwise обучение (BCE/MSE loss).

**Что стало:** Полная поддержка Pairwise обучения с расширяемой системой лоссов.

**Ключевые изменения:**

1. **Pairwise training loop:**
   - Векторизованное формирование пар по ключу
   - `generate_pair_training_batches()` — формирование батчей (запрос, позитив, негатив)
   - Новый тип обучения в train loop с интеграцией online ensembles
   - Модификация `_evaluate()` для pair accuracy

2. **TaskType.PAIRWISE и реестр лоссов:**
   - Расширяемая система регистрации лоссов
   - Встроенные лоссы: `bce_pairwise`, `margin_ranknet`
   - Новый тип задачи в info.json

3. **Online ensembles:**
   - Поддержка онлайн-ансамблей для pairwise режима
   - Корректная интеграция с `update_online_ensembles()`

4. **Convert polars optimizations:**
   - Streaming mode для файлов 50M+ строк
   - `read_csv_batched` с батчами по 500k строк
   - `generate_yql_query.py` для генерации SQL-запросов с ignored-features
   - Сквозной механизм original_keys.tsv для джойна предсказаний

5. **Inference enhancements:**
   - `--metrics` — выгрузка RawFormulaVal с оригинальными ключами
   - `--data-path` — переопределение пути к данным
   - Корректная работа с датасетами, содержащими не все стандартные парты

### 3.3 Ветка `pairwise-lazy` — LazyDataset и оптимизация загрузки

**Что было:** Обучение ограничено размером VRAM.

**Что стало:** Поддержка DataLoader-подобной загрузки данных с pinned memory и prefetch.

**Ключевые изменения:**

1. **Dataset.to_torch()** — параметры `device` и `pin_memory` для контроля размещения данных.

2. **PinnedBatchStager:**
   - Преаллоцированные pinned CPU-буферы под max-форму батча
   - `torch.index_select(src, 0, idx, out=pinned_buf)` вместо advanced indexing
   - Кольцо буферов + CUDA-events для защиты от перезаписи
   - Честный async H2D transfer

3. **Мелкие данные на GPU:**
   - Y_train, pair indices переносятся на GPU один раз
   - pair indices на CPU при `data_on_cpu=True` (убран GPU→CPU хоп)

4. **PrefetchBatchStager:**
   - Background thread для сбора следующего батча параллельно с GPU compute
   - Включён по умолчанию при `data_on_cpu=True`
   - Отключить: `prefetch_batches = false` в конфиге

**Текущее ограничение:** `torch.set_num_threads(1)` делает CPU gather bottleneck-ом. Prefetch даёт ~10% ускорения, но фундаментальная проблема single-thread gather остаётся.

---

## 4. Итоговые результаты

### 4.1 Воспроизведение статьи

| Датасет | TabM (статья) | TabM (ноутбук) | TabPack (ноутбук) | TabPack-cosine (ноутбук) |
|---------|---------------|-----------------|-------------------|--------------------------|
| diamond | 0.1342 ± 0.0017 | 0.1310 ± 0.0007 | 0.1333 ± 0.0005 | — |
| black-friday | 0.6875 ± 0.0015 | 0.6869 ± 0.0004 | 0.6876 ± 0.0003 | — |
| otto | 0.8268 ± 0.0014 | 0.8275 ± 0.0014 | 0.8247 ± 0.0009 | — |

### 4.2 Результаты на реальных данных (Pointwise)

| Метрика | CatBoost | TabPack | TabPack-cosine |
|---------|----------|---------|----------------|
| ROC-AUC | 0.897724 | 0.906136 | **0.910094** |
| Accuracy | 0.821130 | 0.828628 | **0.832616** |
| F1 | 0.809390 | 0.828230 | **0.832268** |

### 4.3 Подбор hyperparameters (Перу, 1M объектов)

Оптимальная конфигурация: **TabPack† с hidden_dim=384**, результат 0.8045.

### 4.4 Что реализовано

| Функциональность | Статус |
|-----------------|--------|
| Воспроизведение результатов статьи | ✅ |
| Обучение в Nirvana (YRec GPU) | ✅ |
| Pointwise обучение (BCE/MSE) | ✅ |
| Pairwise обучение (bce_pairwise, margin_ranknet) | ✅ |
| Онлайн-ансамбли | ✅ |
| Сохранение модели (greedy ensemble) | ✅ |
| Инференс с воспроизводимостью | ✅ |
| Convert polars (80 ГБ → ~1 час) | ✅ |
| DataPreprocessor (fit/transform) | ✅ |
| Оптимизация памяти (40-50% снижение) | ✅ |
| LazyDataset (data_on_cpu) | ✅ (с ограничениями) |
| RerankService | ✅ |
| Offline-replay замер | ✅ |
| Документация (12 MD-файлов) | ✅ |

---

## 5. Точки роста и дальнейшие шаги

### 5.1 Обучение на больших датасетах

**Проблема:** `torch.set_num_threads(1)` + single-thread CPU gather в LazyDataset замедляет обучение в 13 раз при `data_on_cpu=True`.

**Возможные решения:**
- Увеличение числа потоков torch (но риск ухудшения качества из-за шаринга батчей)
- Пересмотр концепции формирования батчей
- Distributed Data Parallelism (DDP) для multi-GPU обучения

### 5.2 Подбор конфигурации TabPack

**Что нужно сделать:**
- Эксперименты с различными конфигами (сейчас используется дефолтный из статьи)
- Подбор оптимального лосса для pairwise (сейчас BCE для pairwise — произвольный выбор)
- Тестирование различных трансформеров на признаках (подозрение, что noisy-quantile вредит качеству)

### 5.3 Замер качества в Metrics

**Проблема:** Сложная конвертация таблички SoY → JSON серпов для загрузки в Metrics.

**Что нужно:** Найти или реализовать кубик для автоматической конвертации.

### 5.4 Эффект масштаба

При обучении на больших датасетах ожидается рост качества TabPack за счёт эффекта масштаба, что должно дать преимущество перед CatBoost.

---

## Приложения

### Список коммитов по веткам

**Ветка `main`:** 30 коммитов (базовая инфраструктура, оптимизация памяти, сохранение моделей, инференс, документация)

**Ветка `pairwise`:** 18 коммитов поверх main (pairwise обучение, online ensembles, convert оптимизация, inference enhancements)

**Ветка `pairwise-lazy`:** 4 коммита поверх pairwise (DataLoader поддержка, PinnedBatchStager, PrefetchBatchStager, оптимизация pair indices)

**Всего:** ~52 коммита за период июнь–август 2026.

### Структура репозитория

```
tabpack-ag/
├── bin/                    # Основные скрипты обучения
│   └── tabpack/
│       ├── tabpack.py      # Главный файл обучения
│       ├── ensemble_checkpoint.py  # Хранение чекпоинтов ансамбля
│       ├── ensemble_utils.py       # Утилиты для ансамблей
│       ├── metrics.py            # Метрики
│       ├── nn.py                 # Нейросетевые архитектуры
│       ├── optim.py              # Оптимизаторы
│       └── predict.py            # Предсказания
├── convert/                # Конвертация данных
│   ├── convert_polars.py   # Основной конвертер (polars)
│   └── generate_yql_query.py  # Генерация SQL-запросов
├── docs/                   # Документация (12 файлов)
├── exp/                    # Результаты экспериментов
├── infer/                  # Инференс
│   ├── infer.py            # Скрипт инференса
│   └── model.pt            # Сохранённая модель
├── lib/                    # Библиотека
│   ├── data.py             # Работа с данными
│   ├── deep.py             # DL утилиты
│   └── experiment.py       # Управление экспериментами
├── reports/                # Отчёты
├── rerank/                 # RerankService и offline-replay
└── configs/                # Конфиги обучения
```

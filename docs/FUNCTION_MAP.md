# FUNCTION_MAP.md — Полная карта функций и зависимостей

Этот документ содержит полную карту всех функций, классов и их зависимостей в проекте TabPack.

---

## Оглавление

1. [bin/tabpack/tabpack.py](#bintabpacktabpackpy) — Главный файл обучения
2. [bin/tabpack/nn.py](#bintabpacknnpy) — Neural Network модули
3. [bin/tabpack/optim.py](#bintabpackoptimpy) — Оптимизаторы
4. [bin/tabpack/ensemble_utils_torch.py](#bintabpackensemble_utilstorchpy) — Алгоритмы ансамблирования
5. [bin/tabpack/ensemble_checkpoint.py](#bintabpackensemble_checkpointpy) — Хранение чекпоинтов
6. [bin/tabpack/metrics.py](#bintabpackmetricspy) — Метрики (NumPy)
7. [bin/tabpack/metrics_torch.py](#bintabpackmetrics_torchpy) — Метрики (Torch)
8. [bin/tabpack/utils.py](#bintabpackutilspy) — Утилиты
9. [bin/tabpack/types.py](#bintabpacktypespy) — Типы
10. [infer/infer.py](#inferinferpy) — Инференс
11. [lib/data.py](#libdatapy) — Работа с данными

---

## bin/tabpack/tabpack.py

### Классы

#### `ModelPack` (строка 58)

**Назначение:** Основная модель-ансамбль, инкапсулирует pack нейронных сетей.

**Компоненты:**
- `num_module` — Embedding для числовых признаков (LinearEmbeddingsPack, CosineEmbeddingsPack и др.)
- `cat_module` — OneHotEncoding для категориальных признаков
- `pack_view` — PackView для преобразования в pack-формат
- `backbone` — MLPBackbonePack (основная нейросеть)
- `output` — LinearPack (выходной слой)

**Зависимости:**
- Импортирует из: `bin.tabpack.nn` (ModulePack, OneHotEncoding, PackView, MLPBackbonePack, LinearPack, LinearEmbeddingsPack, LinearReLUEmbeddingsPack, CosineEmbeddingsPack)
- Использует: `_make_num_module()` (строка 48)

**Методы:**
- `forward()` (строка 107) — Forward pass через все компоненты

**Что влияет:**
- Изменение требует проверки `apply_model_impl()` (строка 158)
- Изменение требует проверки `_evaluate()` (строка 879)
- Изменение требует проверки `infer.py` (build_model_for_ensemble)

---

#### `HyperparameterSampler` (строка 174)

**Назначение:** Обёртка над Optuna для генерации конфигураций моделей.

**Зависимости:**
- Импортирует: `optuna`, `bin.tune._sample_config`
- Использует: `HyperparameterSampler._BASIC_SAMPLERS` (строка 178)

**Методы:**
- `ask()` (строка 218) — Запрашивает новую конфигурацию
- `tell()` (строка 228) — Сообщает результат обучения

**Что влияет:**
- Изменение требует проверки `_prepare_configs()` (строка 1190)
- Изменение требует проверки `_validate_config()` (строка 1174)

---

#### `StatePack` (строка 260)

**Назначение:** Хранит динамическое состояние всех моделей в pack (ещё обучаются).

**Свойства:**
- `ids` — ID каждой модели
- `configs` — Конфигурации гиперпараметров
- `steps` — Текущий шаг обучения
- `n_consequtive_bad_updates` — Счётчик плохих обновлений
- `best_metrics` — Лучшие метрики
- `best_predictions` — Лучшие предсказания
- `best_model_state_dicts` — Лучшие веса

**Зависимости:**
- Импортирует из: `bin.tabpack.nn` (make_keep_pack_idx, ParameterPack, BufferPack, get_pack_size)
- Использует: `_find_arrays_and_tensors()` (строка 245)

**Методы:**
- `validate()` (строка 319) — Проверяет консистентность
- `step()` (строка 340) — Инкрементирует счётчик шагов
- `remove()` (строка 343) — Удаляет завершённые модели
- `update()` (строка 384) — Обновляет best_metrics при улучшении

**Что влияет:**
- Изменение требует проверки `pack_validate()` (строка 486)
- Изменение требует проверки `pack_remove()` (строка 505)
- Изменение требует проверки `compute_stop_pack_idx()` (строка 519)

---

#### `FinalStatePack` (строка 432)

**Назначение:** Хранит результаты моделей, которые уже остановились.

**Свойства:**
- `ids` — ID завершённых моделей
- `steps` — Лучшие шаги завершённых моделей
- `predictions` — Предсказания лучших чекпоинтов (numpy)
- `predictions_torch` — Предсказания лучших чекпоинтов (torch)

**Методы:**
- `extend()` (строка 458) — Добавляет новые завершённые модели

**Что влияет:**
- Изменение требует проверки `main()` (строка 1701)

---

#### `OnlineEnsemble` (строка 603)

**Назначение:** Динамически формирует ансамбль моделей во время обучения.

**Параметры:**
- `type` — Тип алгоритма (greedy, autotopk, bruteforce, beam, topk)
- `update_type` — 'final', 'best' или 'latest' предсказания
- `patience` — Терпение перед остановкой
- `include_current_ensemble_in_pool` — Добавлять ли текущий ансамбль в pool

**Зависимости:**
- Импортирует из: `bin.tabpack.ensemble_utils_torch` (get_ensemble_fn, compute_ensemble_prediction, EnsembleScoreFn)
- Импортирует из: `bin.tabpack.ensemble_utils` (compute_ensemble_prediction)

**Методы:**
- `_prepare_pool()` (строка 678) — Формирует пул кандидатов
- `update()` (строка 749) — Обновляет ансамбль новыми моделями

**Что влияет:**
- Изменение требует проверки `update_online_ensembles()` (строка 802)
- Изменение требует проверки `_make_online_ensembles()` (строка 963)
- Изменение требует проверки ensemble snapshot logic (строка 1803)

---

#### `Config` (строка 1133)

**Назначение:** TypedDict для конфигурации обучения.

**Обязательные поля:**
- `seed`, `data`, `n_models`, `model`, `optimizer`, `batch_size`, `n_epochs`, `patience`

**Опциональные поля:**
- `eval_parts`, `eval_batch_size`, `online_ensembles`, `configs`, `sampler`, `amp_dtype`, `timeout`, `track_experiments`, `track_best_experiment`, `track_online_ensemble_history`, `track_memory_usage`, `save_final_predictions`, `save_all_predictions`, `save_model`

**Что влияет:**
- Изменение требует проверки `_validate_config()` (строка 1174)
- Изменение требует проверки `main()` (строка 1299)
- См. CONFIG.md для полной документации

---

### Функции

#### `_make_num_module()` (строка 48)

**Назначение:** Фабрика для создания num_module (LinearEmbeddingsPack, CosineEmbeddingsPack и др.).

**Зависимости:**
- Импортирует из: `bin.tabpack.nn` (LinearEmbeddingsPack, LinearReLUEmbeddingsPack, CosineEmbeddingsPack)

**Вызывается из:**
- `ModelPack.__init__()` (строка 75)

**Что влияет:**
- Изменение требует проверки `ModelPack` (строка 58)

---

#### `apply_model_impl()` (строка 158)

**Назначение:** Стандартный вызов модели с извлечением данных из Dataset.

**Зависимости:**
- Использует: `lib.data.Dataset`

**Вызывается из:**
- `main()` (строка 1410)
- `_evaluate()` (строка 879)
- `infer.py` (evaluate_ensemble)

**Что влияет:**
- Изменение требует проверки `_evaluate()` (строка 879)
- Изменение требует проверки `infer.py`

---

#### `_find_arrays_and_tensors()` (строка 245)

**Назначение:** Рекурсивно находит все numpy arrays и torch tensors в структуре данных.

**Вызывается из:**
- `StatePack.validate()` (строка 325)

**Что влияет:**
- Изменение требует проверки `StatePack.validate()` (строка 319)

---

#### `pack_validate()` (строка 486)

**Назначение:** Проверяет консистентность model, optimizer и state.

**Зависимости:**
- Импортирует из: `bin.tabpack.nn` (ParameterPack, BufferPack, get_pack_size)

**Вызывается из:**
- `main()` (строка 1555, 1936, 1938)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `pack_remove()` (строка 505)

**Назначение:** Удаляет модели из model, optimizer и state согласованно.

**Зависимости:**
- Импортирует из: `bin.tabpack.nn` (module_pack_remove)
- Импортирует из: `bin.tabpack.optim` (optimizer_pack_remove)

**Вызывается из:**
- `main()` (строка 1742)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- См. GOTCHAS.md#3

---

#### `compute_stop_pack_idx()` (строка 519)

**Назначение:** Определяет какие модели должны остановиться.

**Критерии:**
- `patience` — если `n_consequtive_bad_updates > patience`
- `n_epochs` — если `steps // epoch_size >= n_epochs`

**Вызывается из:**
- `main()` (строка 1658)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `assemble_experiments()` (строка 547)

**Назначение:** Создаёт список ExperimentDict для завершённых моделей.

**Зависимости:**
- Использует: `lib.experiment.Report`

**Вызывается из:**
- `main()` (строка 1707)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `get_experiment_val_score()` (строка 575)

**Назначение:** Извлекает val score из эксперимента.

**Вызывается из:**
- `get_best_experiment()` (строка 590)
- `main()` (строка 1718)

**Что влияет:**
- Изменение требует проверки `get_best_experiment()` (строка 579)

---

#### `get_best_experiment()` (строка 579)

**Назначение:** Находит лучший эксперимент по val score.

**Зависимости:**
- Использует: `get_experiment_val_score()` (строка 575)

**Вызывается из:**
- `main()` (строка 1753)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `update_online_ensembles()` (строка 802)

**Назначение:** Обновляет все онлайн ансамбли и возвращает отчёты.

**Зависимости:**
- Импортирует из: `bin.tabpack.ensemble_utils` (compute_ensemble_prediction)
- Использует: `lib.data.Task.calculate_metrics`

**Вызывается из:**
- `main()` (строка 1771)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `OnlineEnsemble` (строка 603)

---

#### `generate_training_batches()` (строка 851)

**Назначение:** Генерация случайных батчей для одной эпохи.

**Особенность:** Каждая модель получает свой порядок батчей.

**Вызывается из:**
- `main()` (строка 1574)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- См. GOTCHAS.md#19

---

#### `_evaluate()` (строка 879)

**Назначение:** Evaluation модели на всех частях данных.

**Зависимости:**
- Импортирует из: `bin.tabpack.metrics` (calculate_metrics_pack)
- Использует: `lib.util.adjust_gpu_memory_usage` (декоратор)
- Использует: `lib.data.RegressionLabelStats`

**Процесс:**
1. Прогоняет все данные батчами
2. Применяет обратную трансформацию (для регрессии)
3. Применяет sigmoid (binclass) или softmax (multiclass)
4. Вычисляет метрики

**Вызывается из:**
- `main()` (строка 1469, 1630, 1697)
- `infer.py` (evaluate_ensemble)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `infer.py`
- См. GOTCHAS.md#5, #6

---

#### `_free_mps_memory()` (строка 940)

**Назначение:** Освобождает GPU память (MPS и CUDA).

**Вызывается из:**
- `main()` (строка 1348, 1617, 1762, 1765, 1861)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `_make_Y_train()` (строка 948)

**Назначение:** Создаёт тензор Y_train с правильным dtype.

**Вызывается из:**
- `main()` (строка 1351)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `_make_autocast()` (строка 955)

**Назначение:** Создаёт autocast для AMP.

**Ограничение:** Поддерживается только bfloat16.

**Вызывается из:**
- `main()` (строка 1316)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- См. GOTCHAS.md#12

---

#### `_make_online_ensembles()` (строка 963)

**Назначение:** Создаёт словарь OnlineEnsemble из конфигурации.

**Зависимости:**
- Импортирует из: `bin.tabpack.ensemble_utils_torch` (make_emsemble_score_fn)

**Вызывается из:**
- `main()` (строка 1418)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `OnlineEnsemble` (строка 603)

---

#### `_make_muon_scale()` (строка 1003)

**Назначение:** Вычисляет масштаб для Muon optimizer.

**Зависимости:**
- Использует: `bin.tabpack.nn.LinearPack`

**Вызывается из:**
- `main()` (строка 1434)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `_default_zero_weight_decay_condition()` (строка 1026)

**Назначение:** Определяет какие параметры НЕ должны иметь weight_decay.

**Зависимости:**
- Импортирует из: `lib.deep` (default_zero_weight_decay_condition)
- Импортирует из: `bin.tabpack.nn` (LinearEmbeddingsPack)

**Вызывается из:**
- `main()` (строка 1444)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `_make_optimizer()` (строка 1034)

**Назначение:** Фабрика оптимизаторов.

**Поддерживаемые типы:**
- Стандартные: `torch.optim.*` (AdamW, SGD и др.)
- Кастомные: `AdamWPack`, `MuonAdamWPack`

**Особенность:** Автоматически удаляет `pack_size` для стандартных оптимизаторов.

**Вызывается из:**
- `main()` (строка 1441)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- См. GOTCHAS.md#23

---

#### `_make_loss_fn_pack()` (строка 1046)

**Назначение:** Создаёт функцию loss для pack-тензоров.

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

**Вызывается из:**
- `main()` (строка 1463)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- См. GOTCHAS.md#1

---

#### `_get_mean_scores()` (строка 1065)

**Назначение:** Вычисляет средние метрики по всем завершённым моделям.

**Вызывается из:**
- `main()` (строка 1870)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `_get_best_scores()` (строка 1089)

**Назначение:** Вычисляет лучшие метрики среди всех моделей.

**Вызывается из:**
- `main()` (строка 1875)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `_validate_config()` (строка 1174)

**Назначение:** Проверяет корректность конфигурации.

**Проверки:**
- `configs` и `sampler` не могут быть заданы одновременно
- Если `sampler` не в `_BASIC_SAMPLERS`, `n_models` должно быть задано явно

**Вызывается из:**
- `main()` (строка 1300)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `Config` (строка 1133)

---

#### `_prepare_configs()` (строка 1190)

**Назначение:** Подготавливает конфигурации для моделей.

**Логика:**
- Если `configs` задан явно — использует его
- Если `sampler` задан — генерирует через `HyperparameterSampler.ask()`
- Иначе — возвращает None

**Вызывается из:**
- `main()` (строка 1369)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `HyperparameterSampler` (строка 174)

---

#### `_prepare_model_config()` (строка 1204)

**Назначение:** Подготавливает конфигурацию модели с per-model параметрами.

**Логика:**
- Выводит `max_n_blocks`, `max_d_block`, `max_d_embedding` из space или max(value_list)
- Объединяет per-model параметры в model_config

**Зависимости:**
- Импортирует из: `bin.tabpack.utils` (dict_merge_recursively, transpose_list_of_dicts)

**Вызывается из:**
- `main()` (строка 1385)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `ModelPack` (строка 58)

---

#### `main()` (строка 1299)

**Назначение:** Главная функция обучения.

**Полный цикл:**
1. Инициализация (загрузка данных, создание модели, оптимизатора)
2. Training Loop (генерация батчей, forward/backward, evaluation)
3. Stopping Check (early stopping, удаление моделей)
4. Ensemble Update (обновление онлайн ансамблей)
5. Сохранение (checkpoint.pt, model.pt, report.json)

**Зависимости:**
- Импортирует из: `lib.data`, `lib.util`, `lib.experiment`, `lib.deep`, `lib.env`
- Импортирует из: `bin.tabpack.nn`, `bin.tabpack.optim`, `bin.tabpack.metrics`, `bin.tabpack.ensemble_utils`, `bin.tabpack.ensemble_utils_torch`, `bin.tabpack.ensemble_checkpoint`, `bin.tabpack.utils`
- Импортирует из: `bin.tune` (_sample_config)
- Импортирует из: `delu` (random, tools.Timer)

**Вызывает:**
- `_validate_config()` (строка 1300)
- `_prepare_configs()` (строка 1369)
- `_prepare_model_config()` (строка 1385)
- `_make_optimizer()` (строка 1441)
- `_make_loss_fn_pack()` (строка 1463)
- `_evaluate()` (строка 1469, 1630, 1697)
- `generate_training_batches()` (строка 1574)
- `compute_stop_pack_idx()` (строка 1658)
- `pack_remove()` (строка 1742)
- `update_online_ensembles()` (строка 1771)
- `assemble_experiments()` (строка 1707)
- `get_best_experiment()` (строка 1753)
- `_get_mean_scores()` (строка 1870)
- `_get_best_scores()` (строка 1875)

**Что влияет:**
- Изменение требует проверки всех вызываемых функций
- Изменение требует проверки `infer.py` (load_model_artifact, build_model_for_ensemble)
- См. TABPACK_MAIN.md для полной документации

---

## bin/tabpack/nn.py

### Константы

#### `PACK_DIM`, `BATCH_DIM` (строка 19-20)

**Назначение:** Определяют размерности pack и batch в тензорах.

**Значения:**
- `PACK_DIM = 0`
- `BATCH_DIM = 1`

**Используются в:**
- Все pack-модули (LinearPack, MLPBackbonePack и др.)
- `tabpack.py` (apply_model_impl, _evaluate, _make_loss_fn_pack)
- `metrics.py`, `metrics_torch.py`

**Что влияет:**
- Изменение требует проверки ВСЕХ pack-модулей
- См. GOTCHAS.md#2

---

### Классы

#### `OneHotEncoding` (строка 25)

**Назначение:** One-hot кодирование категориальных признаков.

**Особенность:** Поддерживает OOV (out-of-vocabulary) категории — кодирует их как all-zeros.

**Используется в:**
- `ModelPack` (строка 82)

**Что влияет:**
- Изменение требует проверки `ModelPack` (строка 58)
- См. GOTCHAS.md#16

---

#### `PackView` (строка 68)

**Назначение:** Преобразование входных данных в pack-формат.

**Логика:**
- Если вход 2D `(batch, features)` → расширяет до `(pack_size, batch, features)`
- Если вход уже 3D → пропускает без изменений

**Используется в:**
- `ModelPack` (строка 87)

**Что влияет:**
- Изменение требует проверки `ModelPack` (строка 58)

---

#### `ParameterPack` (строка 102)

**Назначение:** Маркерный класс для обучаемых параметров в pack-модулях.

**Наследование:** `nn.Parameter`

**Используется в:**
- Все pack-модули (LinearPack, MLPBackbonePack и др.)
- `OptimizerPack` (строка 65)
- `pack_validate()` (строка 493)

**Что влияет:**
- Изменение требует проверки всех pack-модулей
- Изменение требует проверки `OptimizerPack` (строка 42)

---

#### `BufferPack` (строка 135)

**Назначение:** Маркерный класс для необучаемых буферов (например, размеры слоёв).

**Особенность:** Использует метакласс `_BufferPackMeta` для корректной работы `isinstance()`.

**Используется в:**
- Все pack-модули (LinearPack, MLPBackbonePack и др.)
- `pack_validate()` (строка 493)

**Что влияет:**
- Изменение требует проверки всех pack-модулей

---

#### `ModulePack` (строка 165)

**Назначение:** Базовый абстрактный класс для всех pack-модулей.

**Обязательные свойства:**
- `pack_size` — Возвращает размер pack

**Наследники:**
- `LinearPack` (строка 212)
- `DropoutPack` (строка 352)
- `LeakyReLUPack` (строка 385)
- `MLPBackbonePack` (строка 435)
- `LinearEmbeddingsPack` (строка 586)
- `LinearReLUEmbeddingsPack` (строка 663)
- `CosineEmbeddingsPack` (строка 770)

**Что влияет:**
- Изменение требует проверки всех наследников

---

#### `LinearPack` (строка 212)

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

**Используется в:**
- `MLPBackbonePack.Block` (строка 448)
- `ModelPack.output` (строка 93)

**Что влияет:**
- Изменение требует проверки `MLPBackbonePack` (строка 435)
- Изменение требует проверки `ModelPack` (строка 58)

---

#### `DropoutPack` (строка 352)

**Назначение:** Pack dropout-слоёв с разными вероятностями.

**Параметр:** `p` — float или list[float]

**Используется в:**
- `MLPBackbonePack.Block` (строка 453)

**Что влияет:**
- Изменение требует проверки `MLPBackbonePack` (строка 435)

---

#### `LeakyReLUPack` (строка 385)

**Назначение:** Pack LeakyReLU активаций (демонстрационный пример).

**Параметр:** `negative_slope` — float или list[float]

**Используется в:**
- Не используется в production (демонстрационный пример)

**Что влияет:**
- Изменение не влияет на production код

---

#### `MLPBackbonePack` (строка 435)

**Назначение:** Pack многослойных перцептронов.

**Параметры:**
- `d_in` — размер входа
- `n_blocks` — int или list[int] (количество блоков для каждой модели)
- `d_block` — int или list[int] (размер скрытого слоя)
- `dropout` — float или list[float]
- `activation` — тип активации ('ReLU' и др.)
- `max_n_blocks` — максимум блоков
- `max_d_block` — максимум размера блока

**Внутренний класс `Block` (строка 439):**
- `linear` — LinearPack
- `activation` — Активация
- `dropout` — DropoutPack

**Ключевой метод `_compute_block_idx_list()` (строка 524):**
Вычисляет, какие модели активны в каждом блоке.

**Используется в:**
- `ModelPack.backbone` (строка 88)

**Что влияет:**
- Изменение требует проверки `ModelPack` (строка 58)
- См. GOTCHAS.md#15

---

#### `LinearEmbeddingsPack` (строка 586)

**Назначение:** Линейное embedding числовых признаков.

**Формула:** `output = weight * input + bias`

**Параметры:**
- `n_features` — количество признаков
- `d_embedding` — размер embedding (int или list)
- `max_d_embedding` — максимум для padding

**Используется в:**
- `ModelPack.num_module` (строка 75)
- `_make_num_module()` (строка 48)

**Что влияет:**
- Изменение требует проверки `ModelPack` (строка 58)
- Изменение требует проверки `_default_zero_weight_decay_condition()` (строка 1026)

---

#### `LinearReLUEmbeddingsPack` (строка 663)

**Назначение:** Linear + ReLU embedding.

**Опция `concat_input`:** Конкатенирует исходный признак с embedding.

**Используется в:**
- `ModelPack.num_module` (строка 75)
- `_make_num_module()` (строка 48)

**Что влияет:**
- Изменение требует проверки `ModelPack` (строка 58)

---

#### `CosineEmbeddingsPack` (строка 770)

**Назначение:** Cosine embedding (из статьи TabPack).

**Формула:**
```
1. x = weight * input + bias
2. x = cos(2 * pi * x)
3. x = elementwise_affine(x)
```

**Особенность:** Исходный признак конкатенируется с результатом.

**Используется в:**
- `ModelPack.num_module` (строка 75)
- `_make_num_module()` (строка 48)

**Что влияет:**
- Изменение требует проверки `ModelPack` (строка 58)

---

### Функции

#### `get_pack_size()` (строка 63)

**Назначение:** Получить размер pack dimension тензора.

**Используется в:**
- `_make_loss_fn_pack()` (строка 1054)
- `OptimizerPack` (строка 174)
- `pack_validate()` (строка 494)

**Что влияет:**
- Изменение требует проверки всех мест использования

---

#### `_prepare_dimension_pack()` (строка 191)

**Назначение:** Подготавливает dimension buffer для pack-модулей.

**Используется в:**
- `LinearPack` (строка 212)
- `DropoutPack` (строка 352)
- `MLPBackbonePack` (строка 435)

**Что влияет:**
- Изменение требует проверки всех pack-модулей

---

#### `module_pack_load_state_dict()` (строка 884)

**Назначение:** Загрузка весов в конкретные позиции pack.

**Параметры:**
- `pack_idx` — индексы в целевом pack
- `state_dict_idx` — индексы в source state_dict (по умолчанию те же)

**Используется в:**
- `main()` (строка 1690)
- `infer.py` (load_all_weights)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `infer.py`
- См. GOTCHAS.md#14

---

#### `make_keep_pack_idx()` (строка 902)

**Назначение:** Вычислить индексы для сохранения при удалении моделей.

**Используется в:**
- `StatePack.remove()` (строка 348)
- `main()` (строка 1673)

**Что влияет:**
- Изменение требует проверки `StatePack.remove()` (строка 343)
- Изменение требует проверки `main()` (строка 1299)

---

#### `module_pack_remove()` (строка 910)

**Назначение:** Физически удалить модели из pack.

**Действия:**
1. Создаёт новые ParameterPack/BufferPack без удалённых моделей
2. Обновляет `_pack_size` в PackView
3. Возвращает маппинг old→new параметров

**Используется в:**
- `pack_remove()` (строка 514)

**Что влияет:**
- Изменение требует проверки `pack_remove()` (строка 505)
- См. GOTCHAS.md#3

---

#### `module_pack_select()` (строка 947)

**Назначение:** Контекстный менеджер для временного выбора подмножества моделей.

**Используется в:**
- `main()` (строка 1693)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

## bin/tabpack/optim.py

### Классы

#### `OptimizerPack` (строка 42)

**Назначение:** Базовый класс для pack-оптимизаторов.

**Особенность:** Преобразует списки параметров в тензоры для pack-обработки.

**Наследники:**
- `AdamWPack` (строка 99)
- `MuonAdamWPack` (строка 218)

**Что влияет:**
- Изменение требует проверки всех наследников

---

#### `AdamWPack` (строка 99)

**Назначение:** Pack-версия AdamW с поддержкой разных гиперпараметров.

**Параметры:**
- `lr` — float или list[float] (learning rate)
- `beta1`, `beta2` — моменты
- `eps` — численная стабильность
- `weight_decay` — регуляризация
- `pack_size` — размер pack
- `shared_step` — общий счётчик шагов для всех моделей
- `follow_pytorch` — следовать ли реализации PyTorch

**Используется в:**
- `_make_optimizer()` (строка 1034)

**Что влияет:**
- Изменение требует проверки `_make_optimizer()` (строка 1034)
- Изменение требует проверки `main()` (строка 1299)

---

#### `MuonAdamWPack` (строка 218)

**Назначение:** Гибридный оптимизатор — Muon для линейных слоёв, AdamW для остальных.

**Дополнительные параметры:**
- `muon_lr` — learning rate для Muon
- `muon_momentum` — моментум для Muon
- `muon_ns_steps` — шаги метода Ньютона-Шульца
- `muon_nesterov` — использовать ли Nesterov momentum
- `muon_update_scale` — масштабирование обновления

**Используется в:**
- `_make_optimizer()` (строка 1034)

**Что влияет:**
- Изменение требует проверки `_make_optimizer()` (строка 1034)
- Изменение требует проверки `main()` (строка 1299)

---

### Функции

#### `optimizer_pack_remove()` (строка 416)

**Назначение:** Удалить модели из optimizer state.

**Действия:**
1. Фильтрует tensor-параметры в param_groups
2. Обновляет optimizer state для оставшихся моделей
3. Заменяет старые ParameterPack на новые

**Используется в:**
- `pack_remove()` (строка 515)

**Что влияет:**
- Изменение требует проверки `pack_remove()` (строка 505)
- См. GOTCHAS.md#3

---

## bin/tabpack/ensemble_utils_torch.py

### Типы

#### `EnsembleScoreFn` (строка 20)

**Назначение:** Тип функции оценки ансамбля.

**Сигнатура:** `Callable[[Tensor], Tensor]`

**Используется в:**
- `make_emsemble_score_fn()` (строка 23)
- `OnlineEnsemble` (строка 603)
- Все ensemble алгоритмы

**Что влияет:**
- Изменение требует проверки всех ensemble алгоритмов

---

#### `EnsembleFn` (строка 52)

**Назначение:** Тип функции ансамблирования.

**Сигнатура:** `Callable[..., Tensor | tuple[Tensor, Tensor]]`

**Используется в:**
- Все ensemble алгоритмы
- `get_ensemble_fn()` (строка 434)

**Что влияет:**
- Изменение требует проверки всех ensemble алгоритмов

---

### Функции

#### `make_emsemble_score_fn()` (строка 23)

**Назначение:** Создаёт функцию оценки на основе метрики задачи.

**Зависимости:**
- Импортирует из: `bin.tabpack.metrics_torch` (calculate_metrics_pack)

**Используется в:**
- `_make_online_ensembles()` (строка 971)

**Что влияет:**
- Изменение требует проверки `_make_online_ensembles()` (строка 963)

---

#### `compute_ensemble_prediction()` (строка 77)

**Назначение:** Вычислить предсказание ансамбля.

**Логика:**
- Если weights=None → простое среднее
- Иначе → взвешенное среднее с clamp для probs

**Используется в:**
- `OnlineEnsemble.update()` (строка 767)
- `update_online_ensembles()` (строка 822)
- `infer.py` (evaluate_ensemble)

**Что влияет:**
- Изменение требует проверки `OnlineEnsemble` (строка 603)
- Изменение требует проверки `infer.py`

---

#### `topk_ensemble()` (строка 97)

**Назначение:** Выбирает top-K моделей по score.

**Сложность:** O(N log N)

**Используется в:**
- `get_ensemble_fn()` (строка 434)

**Что влияет:**
- Изменение требует проверки `get_ensemble_fn()` (строка 434)

---

#### `autotopk_ensemble()` (строка 110)

**Назначение:** Автоматически определяет размер ансамбля.

**Алгоритм:**
1. Отсортировать модели по score
2. Постепенно добавлять модели (1, 2, 3, ...)
3. Остановиться когда score перестанет расти

**Используется в:**
- `get_ensemble_fn()` (строка 434)

**Что влияет:**
- Изменение требует проверки `get_ensemble_fn()` (строка 434)

---

#### `bruteforce_ensemble()` (строка 141)

**Назначение:** Перебор всех комбинаций.

**Сложность:** O(C(N,K)) — применим только для малых N

**Используется в:**
- `get_ensemble_fn()` (строка 434)

**Что влияет:**
- Изменение требует проверки `get_ensemble_fn()` (строка 434)

---

#### `greedy_ensemble()` (строка 160)

**Назначение:** Жадный алгоритм с весами.

**Алгоритм:**
1. Инициализация (top-k, autotopk, bruteforce)
2. Итеративное добавление моделей
3. Поддержка `with_replacement` (модель может добавиться несколько раз)
4. Double scoring (опционально)

**Возвращает:** `(ensemble_idx, ensemble_weights)`

**Используется в:**
- `get_ensemble_fn()` (строка 434)

**Что влияет:**
- Изменение требует проверки `get_ensemble_fn()` (строка 434)

---

#### `beam_ensemble()` (строка 339)

**Назначение:** Beam search по комбинациям.

**Параметры:**
- `beam_size` — ширина луча
- `max_ensemble_size` — максимум моделей
- `prefer_minimal_updates` — предпочитать меньшие ансамбли

**Используется в:**
- `get_ensemble_fn()` (строка 434)

**Что влияет:**
- Изменение требует проверки `get_ensemble_fn()` (строка 434)

---

#### `get_ensemble_fn()` (строка 434)

**Назначение:** Фабрика алгоритмов ансамблирования.

**Возвращает:** Одну из функций:
- `topk_ensemble`
- `autotopk_ensemble`
- `bruteforce_ensemble`
- `greedy_ensemble`
- `beam_ensemble`

**Используется в:**
- `OnlineEnsemble.__init__()` (строка 620)

**Что влияет:**
- Изменение требует проверки `OnlineEnsemble` (строка 603)

---

## bin/tabpack/ensemble_checkpoint.py

### Классы

#### `EnsembleCheckpointStore` (строка 19)

**Назначение:** Хранение чекпоинтов по ключу `(model_id, step)`.

**Методы:**
- `save_checkpoint()` (строка 40) — Сохранить
- `get_checkpoint()` (строка 64) — Получить
- `has_checkpoint()` (строка 82) — Проверить
- `remove_checkpoint()` (строка 125) — Удалить
- `get_all_checkpoints()` (строка 94) — Все чекпоинты
- `get_unique_model_ids()` (строка 102) — Уникальные ID

**Используется в:**
- `main()` (строка 1495, 1730, 1819, 1834, 1846, 1850, 1997, 1998, 2037, 2047, 2063, 2064)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- См. GOTCHAS.md#21

---

## bin/tabpack/metrics.py

### Типы

#### `Metrics` (строка 9)

**Назначение:** Тип для метрик (dict[str, np.ndarray]).

**Используется в:**
- `calculate_metrics_pack()` (строка 12)
- `StatePack` (строка 296)
- `_evaluate()` (строка 872)

**Что влияет:**
- Изменение требует проверки всех мест использования

---

### Функции

#### `calculate_metrics_pack()` (строка 12)

**Назначение:** Вычисление метрик для pack-тензоров (NumPy версия).

**Вход:**
- `y_true`: `(samples,)`
- `y_pred`: `(pack_size, samples[, classes])`

**Выход:** `Metrics = dict[str, np.ndarray]` где каждый array имеет форму `(pack_size,)`

**Метрики:**
- Регрессия: rmse, r2
- Binclass: accuracy, roc-auc / cross-entropy
- Multiclass: accuracy, cross-entropy

**Особенность:** y_true replicates до pack_size для векторизации.

**Зависимости:**
- Импортирует из: `lib.data` (Score, _SCORE_HIGHER_IS_BETTER)
- Импортирует из: `lib.types` (PredictionType, TaskType)
- Импортирует из: `bin.tabpack.nn` (BATCH_DIM, PACK_DIM)
- Использует: `sklearn.metrics`

**Используется в:**
- `_evaluate()` (строка 924)

**Что влияет:**
- Изменение требует проверки `_evaluate()` (строка 879)
- Изменение требует проверки `infer.py`

---

## bin/tabpack/metrics_torch.py

### Типы

#### `MetricsTorch` (строка 11)

**Назначение:** Тип для метрик (dict[str, torch.Tensor]).

**Используется в:**
- `calculate_metrics_pack()` (строка 84)

**Что влияет:**
- Изменение требует проверки `calculate_metrics_pack()` (строка 84)

---

### Функции

#### `roc_auc_score()` (строка 14)

**Назначение:** Torch-реализация ROC-AUC (точность до 1e-6 от sklearn).

**Используется в:**
- `calculate_metrics_pack()` (строка 109)

**Что влияет:**
- Изменение требует проверки `calculate_metrics_pack()` (строка 84)

---

#### `multiclass_cross_entropy()` (строка 66)

**Назначение:** Cross-entropy для multiclass без reduction.

**Используется в:**
- `calculate_metrics_pack()` (строка 138)

**Что влияет:**
- Изменение требует проверки `calculate_metrics_pack()` (строка 84)

---

#### `calculate_metrics_pack()` (строка 84)

**Назначение:** Torch-версия метрик для pack.

**Зависимости:**
- Импортирует из: `lib.data` (Score, _SCORE_HIGHER_IS_BETTER)
- Импортирует из: `lib.types` (PredictionType, TaskType)
- Импортирует из: `bin.tabpack.nn` (BATCH_DIM)

**Используется в:**
- `make_emsemble_score_fn()` (строка 33)

**Что влияет:**
- Изменение требует проверки `make_emsemble_score_fn()` (строка 23)

---

## bin/tabpack/utils.py

### Функции

#### `get_path_from_config()` (строка 10)

**Назначение:** Разрешает относительный/абсолютный путь.

**Используется в:**
- Не используется в production коде

**Что влияет:**
- Изменение не влияет на production код

---

#### `time_now()` (строка 18)

**Назначение:** Получить текущее время (time.perf_counter()).

**Используется в:**
- `time_elapsed_since()` (строка 37)

**Что влияет:**
- Изменение требует проверки `time_elapsed_since()` (строка 31)

---

#### `time_elapsed_since()` (строка 31)

**Назначение:** Вычислить время с момента timepoint.

**Зависимости:**
- Использует: `time_now()` (строка 18)

**Используется в:**
- Не используется в production коде

**Что влияет:**
- Изменение не влияет на production код

---

#### `unflatten_dict()` (строка 40)

**Назначение:** Обратное к flatten_dict.

**Используется в:**
- Не используется в production коде

**Что влияет:**
- Изменение не влияет на production код

---

#### `dict_merge_recursively()` (строка 51)

**Назначение:** Рекурсивное слияние словарей.

**Используется в:**
- `_prepare_model_config()` (строка 1284)

**Что влияет:**
- Изменение требует проверки `_prepare_model_config()` (строка 1204)

---

#### `transpose_list_of_dicts()` (строка 66)

**Назначение:** Транспонирование списка словарей.

**Пример:** `[{a:1},{a:2}]` → `{a:[1,2]}`

**Используется в:**
- `main()` (строка 1379, 1390, 1452)
- `_prepare_model_config()` (строка 1289)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `_prepare_model_config()` (строка 1204)

---

#### `to_numpy()` (строка 76)

**Назначение:** Конвертация в numpy.

**Используется в:**
- `main()` (строка 2112, 2115)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `numpy_map()` (строка 102)

**Назначение:** Применить fn ко всем array.

**Используется в:**
- `numpy_index()` (строка 114)

**Что влияет:**
- Изменение требует проверки `numpy_index()` (строка 113)

---

#### `numpy_index()` (строка 113)

**Назначение:** Индексирование numpy структуртур.

**Зависимости:**
- Использует: `numpy_map()` (строка 102)

**Используется в:**
- `ensemble_utils_torch.py` (строка 13)

**Что влияет:**
- Изменение требует проверки `ensemble_utils_torch.py`

---

#### `numpy_stack()` (строка 117)

**Назначение:** Stack по всем array.

**Используется в:**
- `main()` (строка 2111, 2114)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

#### `numpy_concatenate()` (строка 132)

**Назначение:** Concatenate по всем array.

**Используется в:**
- `main()` (строка 2118)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)

---

## bin/tabpack/types.py

### Типы

#### `ConfigDict` (строка 3)

**Назначение:** Тип для конфигураций (dict[str, Any]).

**Используется в:**
- `StatePack` (строка 263)
- `_prepare_configs()` (строка 1190)

**Что влияет:**
- Изменение требует проверки `StatePack` (строка 260)
- Изменение требует проверки `_prepare_configs()` (строка 1190)

---

## infer/infer.py

### Функции

#### `print_nirvana_comparison()` (строка 32)

**Назначение:** Вывести метрики из report.json для сравнения.

**Используется в:**
- `main()` (строка 229)

**Что влияет:**
- Изменение требует проверки `main()` (строка 229)

---

#### `load_model_artifact()` (строка 78)

**Назначение:** Загрузить model.pt с CPU.

**Используется в:**
- `main()` (строка 229)

**Что влияет:**
- Изменение требует проверки `main()` (строка 229)

---

#### `build_model_for_ensemble()` (строка 83)

**Назначение:** Восстановить ModelPack для ансамбля.

**Логика:**
1. Определяет какие ключи в model_config — списки per-model
2. Извлекает только entries для сохранённых model IDs
3. Создаёт ModelPack с правильным pack_size

**Зависимости:**
- Импортирует из: `bin.tabpack.tabpack` (ModelPack)

**Используется в:**
- `main()` (строка 229)

**Что влияет:**
- Изменение требует проверки `main()` (строка 229)
- Изменение требует проверки `ModelPack` (строка 58)

---

#### `load_all_weights()` (строка 117)

**Назначение:** Загрузить веса всех моделей в pack.

**Поддержка форматов:**
- Legacy: ключи — int (model_id)
- Новый: ключи — tuple (model_id, step)

**Зависимости:**
- Импортирует из: `bin.tabpack.nn` (module_pack_load_state_dict)

**Используется в:**
- `main()` (строка 229)

**Что влияет:**
- Изменение требует проверки `main()` (строка 229)
- Изменение требует проверки `module_pack_load_state_dict()` (строка 884)

---

#### `evaluate_ensemble()` (строка 139)

**Назначение:** Оценить ансамбль и усреднить предсказания.

**Параметры:**
- `weights` — веса для взвешенного усреднения (None = простое среднее)

**Зависимости:**
- Импортирует из: `bin.tabpack.tabpack` (_evaluate)
- Импортирует из: `bin.tabpack.ensemble_utils_torch` (compute_ensemble_prediction)
- Импортирует из: `bin.tabpack.metrics` (calculate_metrics_pack)

**Используется в:**
- `main()` (строка 229)

**Что влияет:**
- Изменение требует проверки `main()` (строка 229)
- Изменение требует проверки `_evaluate()` (строка 879)

---

#### `main()` (строка 229)

**Назначение:** Главная функция инференса.

**Поток выполнения:**
1. Загрузка артефакта
2. Построение датасета
3. Создание модели
4. Оценка
5. Сравнение (опционально)

**Зависимости:**
- Импортирует из: `lib.data`, `lib.util`
- Импортирует из: `bin.tabpack.tabpack`, `bin.tabpack.nn`, `bin.tabpack.metrics`

**Что влияет:**
- Изменение требует проверки всех вызываемых функций
- См. INFER_MODULE.md для полной документации

---

## lib/data.py

### Константы

#### `_X_NUM_DTYPE`, `_X_BIN_DTYPE`, `_X_CAT_INT_DTYPE`, `_X_CAT_STR_DTYPE`, `_Y_REG_DTYPE`, `_Y_CLF_DTYPE`, `_SPLIT_DTYPE` (строка 27-39)

**Назначение:** Определяют dtype для различных типов данных.

**Используются в:**
- `load_data()` (строка 492)
- `DataPreprocessor` (строка 41)

**Что влияет:**
- Изменение требует проверки `load_data()` (строка 492)
- Изменение требует проверки `DataPreprocessor` (строка 41)

---

### Классы

#### `DataPreprocessor` (строка 41)

**Назначение:** Препроцессор с паттерном fit/transform для воспроизводимости.

**Параметры конфига:**
- `seed` — случайное зерно
- `extract_bin_from_num` — извлекать ли бинарные из числовых
- `skip_bin_encoder` — пропускать OrdinalEncoder для бинарных
- `num_policy` — 'standard' или 'noisy-quantile'
- `num_memory_efficient` — columnwise трансформация
- `bin_policy` — 'convert-to-cat'
- `cat_policy` — 'ordinal' или 'one-hot'

**Методы:**
- `fit()` (строка 98) — Обучает трансформеры на train данных
- `transform()` (строка 225) — Применяет трансформеры ко всем частям
- `fit_transform()` (строка 366) — Комбинация fit + transform

**Используется в:**
- `main()` (строка 1331, 1340)
- `infer.py` (main)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `infer.py`
- См. GOTCHAS.md#17

---

#### `Task` (строка 847)

**Назначение:** Хранит информацию о задаче.

**Поля:**
- `labels` — метки для каждой части
- `type_` — TaskType (REGRESSION, BINCLASS, MULTICLASS)
- `score` — Score (ACCURACY, ROC_AUC, RMSE и др.)

**Методы:**
- `is_regression` (строка 874) — проверка типа задачи
- `compute_n_classes()` (строка 889) — количество классов
- `calculate_metrics()` (строка 896) — вычисление метрик

**Используется в:**
- `Dataset` (строка 920)
- `update_online_ensembles()` (строка 805)
- `make_emsemble_score_fn()` (строка 24)

**Что влияет:**
- Изменение требует проверки `Dataset` (строка 920)
- Изменение требует проверки `update_online_ensembles()` (строка 802)

---

#### `Score` (строка 828)

**Назначение:** Перечисление доступных метрик.

**Значения:**
- `ACCURACY`
- `CROSS_ENTROPY`
- `MAE`
- `R2`
- `RMSE`
- `ROC_AUC`

**Используется в:**
- `Task` (строка 847)
- `calculate_metrics_pack()` (строка 12, 84)

**Что влияет:**
- Изменение требует проверки `Task` (строка 847)
- Изменение требует проверки `calculate_metrics_pack()` (строка 12, 84)

---

#### `Dataset` (строка 920)

**Назначение:** Dataclass для хранения данных и задачи.

**Поля:**
- `data` — словарь `{data_key: {part_key: array}}`
- `task` — объект Task

**Методы:**
- `from_dir()` (строка 931) — загрузка из директории
- `to_torch()` (строка 937) — конвертация в torch tensors
- `size()` (строка 965) — размер части
- `compute_cat_cardinalities()` (строка 975) — кардинальности категорий
- `try_standardize_labels_()` (строка 1016) — стандартизация для регрессии

**Используется в:**
- `main()` (строка 1334, 1340, 1346)
- `apply_model_impl()` (строка 158)
- `_evaluate()` (строка 879)
- `infer.py` (main)

**Что влияет:**
- Изменение требует проверки `main()` (строка 1299)
- Изменение требует проверки `apply_model_impl()` (строка 158)
- Изменение требует проверки `_evaluate()` (строка 879)
- Изменение требует проверки `infer.py`

---

### Функции

#### `load_data()` (строка 492)

**Назначение:** Загрузить все .npy файлы и применить сплит.

**Используется в:**
- `Dataset.from_dir()` (строка 931)

**Что влияет:**
- Изменение требует проверки `Dataset.from_dir()` (строка 931)

---

#### `load_split()` (строка 450)

**Назначение:** Загрузить индексы сплита из файловой системы.

**Используется в:**
- `apply_split()` (строка 473)

**Что влияет:**
- Изменение требует проверки `apply_split()` (строка 473)

---

#### `apply_split()` (строка 473)

**Назначение:** Разделить данные по частям сплита.

**Зависимости:**
- Использует: `load_split()` (строка 450)

**Используется в:**
- `load_data()` (строка 492)

**Что влияет:**
- Изменение требует проверки `load_data()` (строка 492)

---

#### `transform_num()` (строка 541)

**Назначение:** Трансформация числовых признаков.

**Политики:**
- `STANDARD` — StandardScaler
- `NOISY_QUANTILE` — QuantileTransformer с шумом

**Используется в:**
- `DataPreprocessor.transform()` (строка 225)

**Что влияет:**
- Изменение требует проверки `DataPreprocessor.transform()` (строка 225)

---

#### `transform_cat()` (строка 752)

**Назначение:** Трансформация категориальных признаков.

**Политики:**
- `ORDINAL` — OrdinalEncoder
- `ONE_HOT` — OrdinalEncoder → OneHotEncoder

**Используется в:**
- `DataPreprocessor.transform()` (строка 225)

**Что влияет:**
- Изменение требует проверки `DataPreprocessor.transform()` (строка 225)

---

#### `build_dataset()` (строка 1023)

**Назначение:** Фабричная функция для создания датасета.

**Используется в:**
- `infer.py` (main, fallback для legacy)

**Что влияет:**
- Изменение требует проверки `infer.py`

---

## Диаграмма зависимостей

```
main() (tabpack.py:1299)
├── _validate_config() (tabpack.py:1174)
├── _prepare_configs() (tabpack.py:1190)
│   └── HyperparameterSampler.ask() (tabpack.py:218)
├── _prepare_model_config() (tabpack.py:1204)
│   └── dict_merge_recursively() (utils.py:51)
│   └── transpose_list_of_dicts() (utils.py:66)
├── ModelPack (tabpack.py:58)
│   ├── _make_num_module() (tabpack.py:48)
│   │   └── LinearEmbeddingsPack, CosineEmbeddingsPack (nn.py)
│   ├── OneHotEncoding (nn.py:25)
│   ├── PackView (nn.py:68)
│   ├── MLPBackbonePack (nn.py:435)
│   │   ├── LinearPack (nn.py:212)
│   │   └── DropoutPack (nn.py:352)
│   └── LinearPack (nn.py:212)
├── _make_optimizer() (tabpack.py:1034)
│   └── AdamWPack, MuonAdamWPack (optim.py)
├── _make_loss_fn_pack() (tabpack.py:1046)
├── _evaluate() (tabpack.py:879)
│   └── calculate_metrics_pack() (metrics.py:12)
├── generate_training_batches() (tabpack.py:851)
├── compute_stop_pack_idx() (tabpack.py:519)
├── pack_remove() (tabpack.py:505)
│   ├── module_pack_remove() (nn.py:910)
│   ├── optimizer_pack_remove() (optim.py:416)
│   └── StatePack.remove() (tabpack.py:343)
├── update_online_ensembles() (tabpack.py:802)
│   └── OnlineEnsemble.update() (tabpack.py:749)
│       └── get_ensemble_fn() (ensemble_utils_torch.py:434)
│           └── greedy_ensemble, autotopk_ensemble, etc.
├── assemble_experiments() (tabpack.py:547)
├── get_best_experiment() (tabpack.py:579)
├── _get_mean_scores() (tabpack.py:1065)
├── _get_best_scores() (tabpack.py:1089)
└── EnsembleCheckpointStore (ensemble_checkpoint.py:19)

infer.py main() (infer.py:229)
├── load_model_artifact() (infer.py:78)
├── build_model_for_ensemble() (infer.py:83)
│   └── ModelPack (tabpack.py:58)
├── load_all_weights() (infer.py:117)
│   └── module_pack_load_state_dict() (nn.py:884)
├── evaluate_ensemble() (infer.py:139)
│   ├── _evaluate() (tabpack.py:879)
│   └── compute_ensemble_prediction() (ensemble_utils_torch.py:77)
└── print_nirvana_comparison() (infer.py:32)

lib/data.py
├── DataPreprocessor (data.py:41)
│   ├── fit() (data.py:98)
│   ├── transform() (data.py:225)
│   │   ├── transform_num() (data.py:541)
│   │   └── transform_cat() (data.py:752)
│   └── fit_transform() (data.py:366)
├── Dataset (data.py:920)
│   ├── from_dir() (data.py:931)
│   │   └── load_data() (data.py:492)
│   │       └── apply_split() (data.py:473)
│   │           └── load_split() (data.py:450)
│   ├── to_torch() (data.py:937)
│   └── try_standardize_labels_() (data.py:1016)
├── Task (data.py:847)
└── Score (data.py:828)
```

---

## Критические пути

### Путь 1: Добавление нового loss

1. `_make_loss_fn_pack()` (tabpack.py:1046) — добавить новый loss
2. Проверить что loss возвращает `(pack_size,)` после `mean(BATCH_DIM)`
3. Проверить `loss.sum()` в training loop (tabpack.py:1594)
4. Обновить метрики в `metrics.py` и `metrics_torch.py`
5. Проверить `prediction_type` совместимость

### Путь 2: Добавление нового optimizer

1. Добавить класс в `optim.py`
2. Обновить `_make_optimizer()` (tabpack.py:1034)
3. Проверить поддержку `pack_size` параметра
4. Если optimizer имеет per-model гиперпараметры — добавить в `configs_T` flow
5. Проверить `optimizer_pack_remove()` для корректного удаления моделей

### Путь 3: Добавление нового ensemble алгоритма

1. Добавить функцию в `ensemble_utils_torch.py`
2. Обновить `get_ensemble_fn()` (ensemble_utils_torch.py:434)
3. Проверить что алгоритм возвращает `(ensemble_idx, ensemble_weights)`
4. Обновить `OnlineEnsemble` если нужны новые параметры
5. Проверить сохранение snapshot в training loop

### Путь 4: Изменение модели

1. Обновить `ModelPack` (tabpack.py:58)
2. Проверить что `forward()` возвращает правильную форму
3. Обновить `_prepare_model_config()` если добавлены новые per-model параметры
4. Проверить `module_pack_load_state_dict()` для корректной загрузки весов
5. Обновить `model.pt` формат если изменилась структура

### Путь 5: Изменение данных

1. Обновить `DataPreprocessor` (data.py:41)
2. Проверить `fit()` обучает только на train
3. Проверить `transform()` применяется ко всем частям
4. Проверить `regression_label_stats` сохраняется и используется
5. Проверить `prediction_type` совместимость

---

## Ссылки

- [CONFIG.md](CONFIG.md) — Полная документация конфига
- [GOTCHAS.md](GOTCHAS.md) — Подводные камни и неявные зависимости
- [DATA_FLOW.md](DATA_FLOW.md) — Поток данных через систему
- [TABPACK_MAIN.md](TABPACK_MAIN.md) — Главный файл обучения
- [NN_MODULE.md](NN_MODULE.md) — Neural Network модули
- [DATA_MODULE.md](DATA_MODULE.md) — Работа с данными
- [INFER_MODULE.md](INFER_MODULE.md) — Инференс
- [OPTIM_ENSEMBLE.md](OPTIM_ENSEMBLE.md) — Оптимизаторы и ансамбли

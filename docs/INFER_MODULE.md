# Модуль infer.py — Инференс обученных моделей

## Назначение

Скрипт для загрузки обученной модели из `model.pt` и оценки её предсказаний.

**Важно:** `model.pt` — это **одна модель-ансамбль** (как одна модель CatBoost). При инференсе мы не выбираем "лучшую MLP из ансамбля" — мы прогоняем данные через все MLP в ансамбле и усредняем их предсказания. Результат — предсказание одной модели.

---

## Формат model.pt

Файл `model.pt` содержит:

```python
{
    'state_dicts': {                    # Веса моделей
        (model_id, step): state_dict,   # Новый формат
        # или
        model_id: state_dict,           # Legacy формат
    },
    'n_models': int,                    # Общее количество моделей
    'model_config': dict,               # Конфигурация модели
    'configs': list,                    # Конфигурации всех моделей
    'n_num_features': int,              # Количество числовых признаков
    'cat_cardinalities': list,          # Кардинальности категорий
    'n_classes': int,                   # Количество классов (None для регрессии)
    'prediction_type': str,             # 'labels' или 'probs'
    'regression_label_stats': dict,     # Статистика для денормализации
    'data_config': dict,               # Конфигурация данных
    'ensemble': dict,                   # Информация об ансамбле
    'preprocessor': DataPreprocessor,   # Сохранённый препроцессор
}
```

---

## Функции

### [`load_model_artifact()`](infer/infer.py:78)

**Назначение:** Загрузить model.pt с CPU.

```python
artifact = load_model_artifact('path/to/model.pt')
```

---

### [`build_model_for_ensemble()`](infer/infer.py:83)

**Назначение:** Восстановить ModelPack для ансамбля.

**Логика:**
1. Определяет какие ключи в model_config — списки per-model
2. Извлекает только entries для сохранённых model IDs
3. Создаёт ModelPack с правильным pack_size

**Пример:**
```python
# Если в model_config:
# n_blocks = [1, 2, 3, 1, 2, 3]  # для всех 6 моделей
# А ансамбль использует модели [1, 3, 5]
# То ensemble_model_config['n_blocks'] = [2, 1, 2]
```

---

### [`load_all_weights()`](infer/infer.py:117)

**Назначение:** Загрузить веса всех моделей в pack.

**Поддержка форматов:**
- Legacy: ключи — int (model_id)
- Новый: ключи — tuple (model_id, step)

---

### [`evaluate_ensemble()`](infer/infer.py:139)

**Назначение:** Оценить ансамбль и усреднить предсказания.

**Параметры:**
- `weights` — веса для взвешенного усреднения (None = простое среднее)

**Процесс:**
1. Вызывает `_evaluate()` для получения предсказаний всех моделей
2. Усредняет предсказания по pack dimension
3. Вычисляет метрики для усреднённого предсказания

---

### [`print_nirvana_comparison()`](infer/infer.py:32)

**Назначение:** Вывести метрики из report.json для сравнения.

---

## [`main()`](infer/infer.py:229)

### Аргументы командной строки

| Аргумент | Описание | По умолчанию |
|----------|----------|--------------|
| `--model-path` | Путь к model.pt | `infer/model.pt` |
| `--device` | Устройство (cuda/cpu) | Автоопределение |
| `--parts` | Части для оценки | `val test` |
| `--batch-size` | Размер батча | 32768 |
| `--compare-nirvana` | Сравнить с метриками обучения | False |

### Поток выполнения

1. **Загрузка артефакта**
   - Определение формата state_dicts (legacy vs новый)
   - Извлечение ensemble_info

2. **Построение датасета**
   - Использование сохранённого DataPreprocessor
   - Fallback на build_dataset (legacy)

3. **Создание модели**
   - [`build_model_for_ensemble()`](infer/infer.py:83)
   - [`load_all_weights()`](infer/infer.py:117)

4. **Оценка**
   - [`evaluate_ensemble()`](infer/infer.py:139)
   - Вывод метрик

5. **Сравнение** (опционально)
   - [`print_nirvana_comparison()`](infer/infer.py:32)

---

## Форматы state_dicts

### Legacy формат
```python
state_dicts = {
    0: {...},  # model_id=0
    1: {...},  # model_id=1
    ...
}
```

### Новый формат
```python
state_dicts = {
    (0, 123060): {...},  # (model_id, step)
    (1, 228540): {...},
    (0, 456780): {...},  # Та же модель, другой step
}
```

Новый формат позволяет:
- Сохранять несколько чекпоинтов одной модели
- Точно воспроизводить ансамбль с конкретными шагами

---

## Ensemble info

```python
ensemble = {
    'ids': [0, 1, 3, 5],           # IDs моделей в ансамбле
    'weights': [0.3, 0.2, 0.4, 0.1],  # Веса (опционально)
    'steps': [123060, 228540, ...],    # Шаги обучения
}
```

Если ensemble отсутствует — используется простое среднее всех сохранённых моделей.

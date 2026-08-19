# Система сохранения моделей и инференс - подробный разбор

## Исходная проблема

В начале работы над проектом **не было возможности сохранить обученную модель** и использовать её для предсказаний на новых данных. Все результаты обучения терялись после завершения скрипта.

**Что было:**
- Обучение проходило, метрики считались, но веса моделей не сохранялись
- Не было возможности загрузить модель и сделать инференс
- Результаты экспериментов нельзя было воспроизвести

**Что нужно было:**
- Сохранять веса лучших моделей
- Сохранять информацию об ансамбле (какие модели вошли, их веса)
- Сохранять препроцессор данных для применения к новым данным
- Иметь скрипт для инференса на новых данных

---

## Сложности, с которыми столкнулись

### 1. Ensemble snapshot и update_type

**Проблема:**
В TabPack есть online ансамбли с разными `update_type`:
- `final` - использовать только завершенные модели
- `best` - использовать лучшие чекпоинты работающих моделей
- `latest` - использовать **текущие** веса работающих моделей

Для `update_type='latest'` возникает сложная ситуация:
```
Эпоха 10: ансамбль улучшился, включает модели [3, 7, 12]
          Модель 3 на шаге 1500 (текущие веса)
          Модель 7 на шаге 2300 (текущие веса)
          Модель 12 на шаге 1800 (текущие веса)

Эпоха 11: веса моделей 3, 7, 12 продолжают меняться
          Но ансамбль был построен на весах из эпохи 10!
```

**Решение:**
Нужно сохранять **snapshot весов** в момент улучшения ансамбля, даже если модели продолжают обучаться.

### 2. Координация сохранения весов

**Проблема:**
Веса моделей хранятся в разных местах:
- `state.best_model_state_dicts` - лучшие веса каждой модели
- `model.state_dict()` - текущие веса
- `checkpoint_store` - сохраненные чекпоинты

Нужно правильно выбрать какие веса сохранять в зависимости от:
- Завершена модель или еще обучается
- Какой `update_type` у ансамбля
- Был ли уже сохранен чекпоинт для этой модели

### 3. Удаление старых чекпоинтов

**Проблема:**
Ансамбль может меняться со временем:
```
Эпоха 10: ансамбль = [3, 7, 12]
Эпоха 15: ансамбль = [3, 7, 15, 18]  # модели 12 больше нет
```

Если не удалять старые чекпоинты, память будет расти бесконечно.

### 4. Формат сохранения state_dicts

**Проблема:**
Изначально state_dicts сохранялись как:
```python
state_dicts = {
    0: {...},  # model_id -> state_dict
    1: {...},
}
```

Но для ансамбля нужно знать **на каком шаге** была модель:
```python
state_dicts = {
    (0, 123060): {...},  # (model_id, step) -> state_dict
    (1, 228540): {...},
}
```

### 5. Воспроизводимость препроцессинга

**Проблема:**
DataPreprocessor обучает трансформеры на train данных:
```python
preprocessor.fit(train_data)  # QuantileTransformer, StandardScaler, etc.
```

При инференсе нужно применить **те же самые** трансформеры к новым данным. Если не сохранить препроцессор, результаты будут невоспроизводимы.

---

## Что было изменено

### Итерация 1: Базовое сохранение (коммиты 2bdda3b, ee3d4ad)

**Добавлено:**
- Сохранение информации об ансамбле в `model.pt`:
  ```python
  torch.save({
      'ensemble': {
          'ids': [3, 7, 12],
          'weights': [0.4, 0.3, 0.3],
          'steps': [1500, 2300, 1800],
      },
      ...
  }, 'model.pt')
  ```

- Скрипт инференса `infer/infer.py`
- Сохранение индексов признаков для правильного порядка

**Проблемы:**
- Веса не сохранялись, только информация об ансамбле
- Нельзя было загрузить модель

### Итерация 2: Сохранение весов (коммиты 72f003c, 79997ab, 833b720)

**Добавлено:**
- Сохранение весов всех моделей из ансамбля:
  ```python
  state_dicts = {}
  for idx, model_id in enumerate(ensemble_ids):
      step = ensemble_steps[idx]
      state_dicts[(model_id, step)] = get_weights(model_id, step)
  ```

**Проблемы:**
- Для `update_type='latest'` сохранялись **лучшие** веса, а не текущие
- Ансамбль был построен на текущих весах, но сохранялись лучшие
- Результаты инференса не совпадали с результатами обучения

### Итерация 3: Исправление для update_type='latest' (коммиты 6f8aa8d, c08ea78)

**Добавлено:**
- Сохранение **текущих** весов для `update_type='latest'`:
  ```python
  if update_type == 'latest':
      state_dict_to_use = model.state_dict()  # Текущие веса
  else:
      state_dict_to_use = state.best_model_state_dicts  # Лучшие веса
  ```

- Сохранение snapshot при каждом улучшении ансамбля:
  ```python
  if first_online_ensemble_improved and save_model:
      # Сохраняем snapshot весов в момент улучшения
      for idx, model_id in enumerate(ensemble_ids):
          save_checkpoint(model_id, step, state_dict_to_use)
  ```

**Проблемы:**
- Старые чекпоинты не удалялись
- Память росла бесконечно

### Итерация 4: EnsembleCheckpointStore (коммит 53dc82f)

**Добавлено:**
- Новый класс для управления чекпоинтами:
  ```python
  class EnsembleCheckpointStore:
      def __init__(self):
          self._checkpoints = {}  # (model_id, step) -> state_dict
      
      def save_checkpoint(self, model_id, step, state_dict):
          self._checkpoints[(model_id, step)] = state_dict
      
      def get_checkpoint(self, model_id, step):
          return self._checkpoints[(model_id, step)]
      
      def has_checkpoint(self, model_id, step):
          return (model_id, step) in self._checkpoints
      
      def remove_checkpoint(self, model_id, step):
          del self._checkpoints[(model_id, step)]
  ```

- Удаление старых чекпоинтов:
  ```python
  # После улучшения ансамбля
  current_ensemble_keys = {(id, step) for id, step in zip(ensemble_ids, ensemble_steps)}
  
  keys_to_remove = [
      key for key in checkpoint_store.get_all_checkpoints()
      if key not in current_ensemble_keys
  ]
  
  for key in keys_to_remove:
      checkpoint_store.remove_checkpoint(*key)
  ```

**Результат:**
- Память больше не растет бесконечно
- Хранятся только чекпоинты текущего ансамбля

### Итерация 5: Сохранение препроцессора (коммит 9f87df3)

**Добавлено:**
- Сохранение обученного DataPreprocessor в `model.pt`:
  ```python
  torch.save({
      'preprocessor': preprocessor,
      'data_config': data_config,
      ...
  }, 'model.pt')
  ```

- Загрузка и применение при инференсе:
  ```python
  artifact = torch.load('model.pt')
  preprocessor = artifact['preprocessor']
  
  # Применяем те же трансформеры к новым данным
  new_dataset = preprocessor.transform(new_dataset)
  ```

**Результат:**
- Полная воспроизводимость результатов
- Препроцессинг идентичен обучению

### Итерация 6: Оптимизация и очистка (коммиты ea57b13, ccc71ce)

**Добавлено:**
- Удален мусорный код
- Убраны лишние глобальные переменные
- Упрощена логика сохранения

---

## Финальная архитектура

### Структура model.pt

```python
{
    # Веса моделей
    'state_dicts': {
        (model_id, step): state_dict,
        (3, 1500): {...},
        (7, 2300): {...},
        (12, 1800): {...},
    },
    
    # Информация об ансамбле
    'ensemble': {
        'ids': [3, 7, 12],
        'weights': [0.4, 0.3, 0.3],
        'steps': [1500, 2300, 1800],
    },
    
    # Конфигурация модели
    'model_config': {
        'n_blocks': [2, 3, 2],  # per-model параметры
        'dropout': [0.1, 0.2, 0.1],
        'd_block': 384,
        ...
    },
    
    # Данные
    'n_num_features': 150,
    'cat_cardinalities': [10, 20, 15],
    'n_classes': None,  # для регрессии
    
    # Препроцессинг
    'preprocessor': DataPreprocessor(...),
    'data_config': {...},
    
    # Прочее
    'prediction_type': 'probs',
    'regression_label_stats': None,
    'n_models': 64,
    'configs': [...],
}
```

### Процесс сохранения

```python
# 1. Инициализация
checkpoint_store = EnsembleCheckpointStore()

# 2. При остановке модели
if save_model:
    for i in stop_pack_idx:
        model_id = state.ids[i]
        step = state.best_steps[i]
        checkpoint_store.save_checkpoint(
            model_id, step, state.best_model_state_dicts[i]
        )

# 3. При улучшении ансамбля
if first_online_ensemble_improved and save_model:
    ensemble_ids = first_ensemble.ids.tolist()
    ensemble_steps = first_ensemble.steps.tolist()
    
    for idx, model_id in enumerate(ensemble_ids):
        step = ensemble_steps[idx]
        
        if not checkpoint_store.has_checkpoint(model_id, step):
            # Определяем какие веса сохранять
            if update_type == 'latest':
                state_dict = model.state_dict()  # Текущие
            else:
                state_dict = state.best_model_state_dicts  # Лучшие
            
            checkpoint_store.save_checkpoint(model_id, step, state_dict)
    
    # Удаляем старые чекпоинты
    current_keys = {(id, step) for id, step in zip(ensemble_ids, ensemble_steps)}
    for key in checkpoint_store.get_all_checkpoints():
        if key not in current_keys:
            checkpoint_store.remove_checkpoint(*key)

# 4. Финальное сохранение
torch.save({
    'state_dicts': {
        key: checkpoint_store.get_checkpoint(*key)
        for key in current_keys
    },
    'ensemble': ensemble_info,
    'preprocessor': preprocessor,
    ...
}, 'model.pt')
```

### Процесс инференса

```python
# 1. Загрузка
artifact = torch.load('model.pt')
state_dicts = artifact['state_dicts']
ensemble_info = artifact['ensemble']
preprocessor = artifact['preprocessor']

# 2. Построение модели
model = build_model_for_ensemble(artifact, ensemble_info['ids'])

# 3. Загрузка весов
load_all_weights(model, state_dicts, ensemble_info['ids'], ensemble_info['steps'])

# 4. Препроцессинг новых данных
new_dataset = preprocessor.transform(new_dataset)

# 5. Инференс
predictions = evaluate_ensemble(
    model, new_dataset, ensemble_info['weights']
)
```

---

## Результаты

### Что работает сейчас

✅ **Полное сохранение модели:**
- Веса всех моделей из ансамбля
- Информация об ансамбле (индексы, веса, шаги)
- Препроцессор данных
- Конфигурация модели

✅ **Воспроизводимость:**
- Результаты инференса совпадают с результатами обучения
- Препроцессинг идентичен обучению
- Для `update_type='latest'` сохраняются правильные веса

✅ **Эффективное использование памяти:**
- Старые чекпоинты удаляются
- Хранятся только чекпоинты текущего ансамбля

✅ **Удобный инференс:**
```bash
python infer/infer.py \
    --model-path exp/tabpack/0/peru/main/model.pt \
    --parts val test \
    --compare-nirvana
```

### Пример вывода инференса

```
=== Ensemble Results ===
  [val]
    pair_accuracy:  0.8339
  [test]
    pair_accuracy:  0.8230

=== Nirvana Comparison ===
  [val]
    Training: 0.8341
    Inference: 0.8339
    Diff: -0.0002
```

---

## Выводы

Система сохранения моделей прошла через **6 итераций** разработки:

1. **Базовое сохранение** - только информация об ансамбле
2. **Сохранение весов** - добавлены state_dicts
3. **Исправление для latest** - сохранение текущих весов
4. **EnsembleCheckpointStore** - управление чекпоинтами
5. **Сохранение препроцессора** - воспроизводимость
6. **Оптимизация** - очистка кода

**Ключевые уроки:**
- Для `update_type='latest'` критически важно сохранять snapshot весов в момент улучшения
- Старые чекпоинты нужно удалять, иначе память растет бесконечно
- Препроцессор нужно сохранять для воспроизводимости
- Формат `(model_id, step)` более гибкий, чем просто `model_id`

**Результат:**
Полноценная система сохранения/загрузки моделей с полной воспроизводимостью результатов.

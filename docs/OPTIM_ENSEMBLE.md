# Модули optim.py и ensemble_utils_torch.py

## optim.py — Оптимизаторы для Pack

### [`OptimizerPack`](bin/tabpack/optim.py:42)

**Назначение:** Базовый класс для pack-оптимизаторов.

**Особенность:** Преобразует списки параметров в тензоры для pack-обработки.

---

### [`AdamWPack`](bin/tabpack/optim.py:99)

**Назначение:** Pack-версия AdamW с поддержкой разных гиперпараметров.

**Параметры:**
- `lr` — float или list[float] (learning rate)
- `beta1`, `beta2` — моменты
- `eps` — численная стабильность
- `weight_decay` — регуляризация
- `pack_size` — размер pack
- `shared_step` — общий счётчик шагов для всех моделей
- `follow_pytorch` — следовать ли реализации PyTorch

**[`step()`](bin/tabpack/optim.py:136):**

Для каждого параметра:
1. Инициализация state (step, exp_avg, exp_avg_sq)
2. Weight decay: `p *= (1 - lr * weight_decay)`
3. Update exp_avg: `exp_avg.lerp_(grad, 1 - beta1)`
4. Update exp_avg_sq: `exp_avg_sq.lerp_(grad², 1 - beta2)`
5. Bias correction
6. Update: `p -= lr * exp_avg / (sqrt(exp_avg_sq) / sqrt(bc2) + eps)`

---

### [`MuonAdamWPack`](bin/tabpack/optim.py:218)

**Назначение:** Гибридный оптимизатор — Muon для линейных слоёв, AdamW для остальных.

**Дополнительные параметры:**
- `muon_lr` — learning rate для Muon
- `muon_momentum` — моментум для Muon
- `muon_ns_steps` — шаги метода Ньютона-Шульца
- `muon_nesterov` — использовать ли Nesterov momentum
- `muon_update_scale` — масштабирование обновления

**[`step()`](bin/tabpack/optim.py:396):**

```python
for group in param_groups:
    if group['muon']:
        self._step_muon(group)  # Для linear.weight
    else:
        self._step_adamw(group)  # Для bias, dropout и др.
```

**Muon step:**
1. Momentum buffer update
2. Gradient sign via Newton-Schulz iteration (zeropower)
3. Scale by `sqrt(max(1, m/n))`
4. Update: `p -= muon_lr * update`

---

### [`optimizer_pack_remove()`](bin/tabpack/optim.py:416)

**Назначение:** Удалить модели из optimizer state.

**Действия:**
1. Фильтрует tensor-параметры в param_groups
2. Обновляет optimizer state для оставшихся моделей
3. Заменяет старые ParameterPack на новые

---

## ensemble_utils_torch.py — Алгоритмы ансамблирования

### Базовые типы

```python
EnsembleScoreFn = Callable[[Tensor], Tensor]  # Функция оценки
EnsembleFn = Callable[..., Tensor | tuple[Tensor, Tensor]]  # Алгоритм
```

---

### [`make_emsemble_score_fn()`](bin/tabpack/ensemble_utils_torch.py:23)

Создаёт функцию оценки на основе метрики задачи.

---

### [`compute_ensemble_prediction()`](bin/tabpack/ensemble_utils_torch.py:77)

**Назначение:** Вычислить предсказание ансамбля.

**Логика:**
- Если weights=None → простое среднее
- Иначе → взвешенное среднее с clamp для probs

---

## Алгоритмы

### [`topk_ensemble()`](bin/tabpack/ensemble_utils_torch.py:97)

**Алгоритм:**
1. Оценить каждую модель
2. Взять top-K по score

**Сложность:** O(N log N)

---

### [`autotopk_ensemble()`](bin/tabpack/ensemble_utils_torch.py:110)

**Алгоритм:**
1. Отсортировать модели по score
2. Постепенно добавлять модели (1, 2, 3, ...)
3. Остановиться когда score перестанет расти

**Результат:** Оптимальный размер ансамбля + состав

---

### [`bruteforce_ensemble()`](bin/tabpack/ensemble_utils_torch.py:141)

**Алгоритм:**
1. Перебрать все комбинации размера K
2. Для каждой: среднее предсказаний → score
3. Вернуть лучшую

**Сложность:** O(C(N,K)) — применим только для малых N

---

### [`greedy_ensemble()`](bin/tabpack/ensemble_utils_torch.py:160)

**Алгоритм:**

1. **Инициализация:**
   - `init_top_k` — взять top-K моделей
   - `init_autotopk` — использовать autotopk
   - `init_bruteforce_k` — bruteforce для начального набора

2. **Итеративное добавление:**
   - Для каждой кандидата: вычислить score ансамбля + кандидат
   - Выбрать лучшего кандидата
   - Если нет улучшения — остановиться
   - Поддержать `with_replacement` (модель может добавиться несколько раз)

3. **Double scoring** (опционально):
   - `score_fn` — основная метрика
   - `selection_score_fn` — метрика для выбора
   - Фильтровать кандидатов по основной, выбирать по secondary

**Возвращает:** `(ensemble_idx, ensemble_weights)`

---

### [`beam_ensemble()`](bin/tabpack/ensemble_utils_torch.py:339)

**Алгоритм:**

1. На каждой итерации рассмотреть добавление 1..beam_size моделей
2. Для каждого размера: перебрать комбинации
3. Выбрать лучшую комбинацию
4. Если нет улучшения — остановиться

**Параметры:**
- `beam_size` — ширина луча
- `max_ensemble_size` — максимум моделей
- `prefer_minimal_updates` — предпочитать меньшие ансамбли

---

### [`get_ensemble_fn()`](bin/tabpack/ensemble_utils_torch.py:434)

Фабрика алгоритмов:
```python
{
    'topk': topk_ensemble,
    'autotopk': autotopk_ensemble,
    'bruteforce': bruteforce_ensemble,
    'greedy': greedy_ensemble,
    'beam': beam_ensemble,
}
```

---

## Online Ensembles

### [`_OnlineEnsemble`](bin/tabpack/ensemble_utils_torch.py:459)

Базовый класс с окном истории.

### [`SimpleOnlineEnsemble`](bin/tabpack/ensemble_utils_torch.py:488)

Простое среднее текущих предсказаний.

### [`_SelectiveOnlineEnsemble`](bin/tabpack/ensemble_utils_torch.py:502)

Использует алгоритм выбора из окна предсказаний.

**Подклассы:**
- [`TopKOnlineEnsemble`](bin/tabpack/ensemble_utils_torch.py:561)
- [`BruteForceOnlineEnsemble`](bin/tabpack/ensemble_utils_torch.py:566)
- [`GreedyOnlineEnsemble`](bin/tabpack/ensemble_utils_torch.py:571)
- [`BeamOnlineEnsemble`](bin/tabpack/ensemble_utils_torch.py:576)

---

## Сравнение алгоритмов

| Алгоритм | Сложность | Качество | Поддержка весов |
|----------|-----------|----------|-----------------|
| topk | O(N log N) | Базовое | Нет |
| autotopk | O(N log N) | Среднее | Нет |
| bruteforce | O(C(N,K)) | Оптимальное | Нет |
| greedy | O(N²) | Хорошее | Да (with_replacement) |
| beam | O(beam·C(N,k)) | Очень хорошее | Нет |

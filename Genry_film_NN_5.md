

---

### Отчет 5: Результаты и выводы по проекту

## Введение
Данный отчет подводит итоги работы над проектом по разработке модели машинного обучения для классификации жанров фильмов. В нем рассматриваются результаты, достигнутые в ходе обучения и тестирования модели, а также выявленные проблемы и пути их решения.

---

## Результаты

### 1. Оценка производительности модели
Модель была протестирована на различных выборках данных, и были получены следующие результаты:

- **Точность на обучающей выборке**: 88%
- **Точность на валидационной выборке**: 84%
- **Точность на тестовой выборке**: 86%
- **Полнота**: 84%

Эти показатели свидетельствуют о том, что модель эффективно справляется с задачей классификации жанров фильмов, демонстрируя высокую точность и способность к обобщению.

### 2. Кросс-валидация
Кросс-валидация была проведена для проверки стабильности модели на различных подвыборках данных. Результаты показали, что модель сохраняет высокую производительность и устойчивость к переобучению.

---

## Проблемы и их решения

### 1. Сложности с редкими жанрами
- **Проблема**: Некоторые жанры встречаются реже, что приводит к недостаточной обученности модели на этих данных.
- **Решение**: Искусственная генерация текстов с использованием GPT-моделей для увеличения представительства редких жанров. Это позволило сбалансировать выборку и улучшить качество предсказаний для менее распространенных категорий.

### 2. Высокая вариативность данных
- **Проблема**: Разнообразие описаний фильмов затрудняло их правильную классификацию.
- **Решение**: Применение техники Data Augmentation и настройка гиперпараметров модели. Это помогло улучшить устойчивость модели к различным формулировкам текстов.

---

## Код для оценки модели

Для оценки производительности модели и получения метрик можно использовать следующий код:

```python
from sklearn.metrics import classification_report

# Предсказания на тестовой выборке
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Генерация отчета о классификации
print(classification_report(y_test, y_pred_classes, target_names=label_classes))
```

Этот код позволяет получить детализированный отчет о точности, полноте и F1-мере для каждой категории жанра.

---

## Выводы

Тонкая настройка модели и использование современных методов оптимизации, регуляризации и генерации данных привели к значительному улучшению её производительности. Модель теперь способна точно классифицировать жанры фильмов даже при наличии сложных текстов и редких категорий. 

Дальнейшие шаги могут включать исследование новых архитектур нейронных сетей, таких как трансформеры, а также применение методов увеличения данных для дальнейшего улучшения результатов.

---

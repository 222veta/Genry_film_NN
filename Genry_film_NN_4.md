

### Отчет 4: Тонкая настройка модели

## Введение
Тонкая настройка модели является ключевым этапом в процессе разработки системы машинного обучения. Этот этап включает в себя применение различных методов для улучшения качества предсказаний модели классификации жанров фильмов. Основное внимание уделяется регуляризации, оптимизации, изменению скорости обучения и увеличению данных.

---

### 1. Регуляризация
Регуляризация — это метод, позволяющий уменьшить переобучение модели, что особенно важно при работе с ограниченными данными. Для этого в модель был добавлен слой **Dropout**:
- **Dropout** отключает случайную долю нейронов во время обучения, что помогает избежать избыточной зависимости от отдельных нейронов и способствует лучшей обобщающей способности модели.

#### Применение Dropout:
- Установлен уровень Dropout на 0.5, что означает, что 50% нейронов будут случайно отключены на каждой итерации обучения.

---

### 2. Оптимизация
Для повышения эффективности обновления весов модели был выбран оптимизатор **AdamW**:
- **AdamW** — это модификация стандартного Adam, которая включает весовую регуляризацию (Weight Decay). Это позволяет улучшить устойчивость модели к переобучению и обеспечивает более стабильное обновление весов.

#### Преимущества AdamW:
- Адаптивное изменение скорости обучения.
- Устойчивость к переобучению за счет регуляризации.

---

### 3. Динамическое изменение скорости обучения
Для более эффективного обучения была реализована функция **LearningRateScheduler**:
- Эта функция адаптирует скорость обучения в зависимости от текущей эпохи:
  - На ранних стадиях обучения скорость обучения остается высокой для быстрого нахождения оптимального решения.
  - На более поздних стадиях скорость обучения снижается, что позволяет модели более точно подстраивать свои веса.

#### Код для изменения скорости обучения:
```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 0.5
```

---

### 4. Увеличение данных (Data Augmentation)
Для повышения разнообразия обучающего набора были использованы GPT-модели для генерации дополнительных описаний фильмов:
- Это позволило увеличить количество уникальных текстов, что сделало модель более устойчивой к различным формулировкам и улучшило её способность к обобщению.

#### Применение GPT для генерации данных:
- Сгенерированные тексты включают описания редких жанров, что помогает сбалансировать выборку.

---

### Код реализации модели

```python
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Определение модели
model = Sequential([
    Embedding(input_dim=len(vocabulary), output_dim=128, input_length=max_sequence_length),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Компиляция модели с AdamW
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели с использованием LearningRateScheduler
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[LearningRateScheduler(scheduler)]
)
```

---

### Результаты

#### Оценка на тестовой выборке
1. **Точность**: 86%
2. **Полнота**: 84%

#### Кросс-валидация
Показала стабильные результаты при проверке на различных подвыборках, что свидетельствует о хорошей обобщающей способности модели.

---

### Проблемы и их решения

1. **Сложности с редкими жанрами**: Искусственная генерация текстов с использованием GPT-моделей для увеличения представительства редких жанров.
2. **Высокая вариативность данных**: Применение техники Data Augmentation и настройка гиперпараметров модели.

---

### Выводы

Тонкая настройка модели и использование современных методов оптимизации, регуляризации и генерации данных привели к значительному улучшению её производительности. Модель теперь способна точно классифицировать жанры фильмов даже при наличии сложных текстов и редких категорий.


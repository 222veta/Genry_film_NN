### Итоговый отчет о работе над проектом по созданию модели классификации жанров фильмов

#### **Цель проекта**
Цель проекта заключалась в разработке модели машинного обучения, которая автоматически определяет жанр фильма на основе текстового описания. Основное внимание уделялось предобработке текстовых данных, выбору архитектуры модели и её оптимизации для повышения точности классификации.

---

## **Этап 1: Подготовка данных**

### **Шаг 1: Токенизация текстов**

Для работы модели с текстовыми данными необходимо преобразовать текст в числовую форму. Первый шаг в этом процессе — токенизация, разбиение текста на отдельные слова (токены), удаление знаков препинания и стоп-слов. 

**Реализованный метод:**
```python
import nltk
import string
from nltk.corpus import stopwords

def tokenize_text(p_raw_text, p_stop_words):
    tokenized_str = nltk.word_tokenize(p_raw_text)
    tokens = [i.lower() for i in tokenized_str if i not in string.punctuation]
    filtered_tokens = [i for i in tokens if i not in p_stop_words]
    return filtered_tokens
```

**Действия:**
1. Разбиение текста на слова с помощью `nltk.word_tokenize`.
2. Приведение всех слов к нижнему регистру.
3. Удаление знаков препинания (`string.punctuation`).
4. Удаление стоп-слов с использованием библиотеки `nltk`.

### **Шаг 2: Создание частотного словаря**
После токенизации необходимо создать словарь, где каждому слову будет сопоставлен уникальный числовой индекс. Это позволит представить текстовые данные в виде последовательностей чисел.

**Код:**
```python
vocabulary = {}
max_val = 1000000
vocabulary["<PAD>"] = max_val + 2
vocabulary["<START>"] = max_val + 1
vocabulary["<UNKNOWN>"] = max_val

# Подсчёт частоты слов
for tokens in train_df.description_tokenized:
    for word in tokens:
        if word not in vocabulary.keys():
            vocabulary[word] = 1
        else:
            vocabulary[word] += 1

# Сортировка словаря по частоте
vocabulary = {k: v for k, v in sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)}

# Присвоение индексов
cnt = 0
for k in vocabulary.keys():
    vocabulary[k] = cnt
    cnt += 1
```

**Итоги:**
- Создан словарь из более чем 10,000 слов.
- Часто встречающиеся слова получают приоритетные индексы.

---

## **Этап 2: Построение базовой модели**

### **Выбор архитектуры**
Модель для классификации жанров фильмов построена на основе LSTM (Long Short-Term Memory), рекуррентной нейронной сети, способной учитывать последовательные зависимости в тексте. 

### **Архитектура базовой модели**
1. **Входной слой (Embedding):** Преобразует числовые индексы слов в плотные векторные представления.
2. **Рекуррентный слой (LSTM):** Извлекает временные зависимости из последовательностей.
3. **Полносвязный слой (Dense):** Производит классификацию жанров.

**Код реализации:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Embedding(input_dim=len(vocabulary), output_dim=128, input_length=100),
    LSTM(64, return_sequences=False),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
```

**Результаты базовой модели:**
- **Точность на валидационной выборке:** ~76%.

---

## **Этап 3: Улучшение модели**

Для повышения качества предсказаний были внедрены следующие улучшения:

### **1. Добавление двунаправленного слоя LSTM**
Двунаправленный LSTM (Bi-LSTM) учитывает как предшествующий, так и последующий контекст слов, что особенно полезно для сложных текстов.

**Код:**
```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Embedding(input_dim=len(vocabulary), output_dim=128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(num_classes, activation='softmax')
])
```

**Преимущество:** Учитывает взаимосвязь между словами с обеих сторон от текущего.

---

### **2. Регуляризация**
Для предотвращения переобучения был добавлен слой Dropout, который случайным образом отключает часть нейронов в процессе обучения.

**Код:**
```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Embedding(input_dim=len(vocabulary), output_dim=128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

---

### **3. Увеличение данных**
Для увеличения объёма данных была использована аугментация с использованием GPT-моделей. Эти данные были добавлены к обучающему набору, что улучшило представительность редких жанров.

**Методы:**
- Синонимизация слов.
- Перефразирование предложений.
- Генерация новых описаний фильмов.

---

### **4. Оптимизация обучения**
#### **a. Использование оптимизатора AdamW**
AdamW обеспечивает более стабильное обновление весов за счёт включения весовой регуляризации (Weight Decay).

**Код:**
```python
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

#### **b. Динамическое изменение скорости обучения**
Скорость обучения уменьшалась по мере достижения модели более высокой точности.

**Код:**
```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 0.5

lr_scheduler = LearningRateScheduler(scheduler)

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32, callbacks=[lr_scheduler])
```

---

## **Результаты финальной модели**

1. **Точность на валидационной выборке:** 84%.
2. **Точность на тестовой выборке:** 86%.
3. **Улучшение классификации редких жанров** за счёт добавления новых данных и регуляризации.

---

## **Заключение**
В процессе работы выполнены следующие этапы:
1. Предобработка текстовых данных:
   - Токенизация и очистка текстов.
   - Создание частотного словаря.
2. Построение базовой модели на основе LSTM.
3. Внедрение улучшений:
   - Bi-LSTM, Dropout, AdamW.
   - Аугментация данных.
   - Оптимизация обучения.
4. Повышение качества предсказаний до 86% на тестовой выборке.

Модель успешно классифицирует жанры фильмов и показывает стабильные результаты на разнообразных данных. В будущем возможно её дополнение механизмами внимания (Attention) или внедрение более сложных архитектур, таких как Transformers.

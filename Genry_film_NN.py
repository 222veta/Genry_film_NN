def tokenize_text (p_raw_text, p_stop_words):
    '''Функция для токенизации текста

    :param p_raw_text: исходная текстовая строка
    :param p_stop_words: список стоп слов
    '''
    
    tokenized_str = nltk.word_tokenize(p_raw_text)
    tokens = [i.lower() for i in tokenized_str if ( i not in string.punctuation )]
    filtered_tokens = [i for i in tokens if ( i not in p_stop_words )]
    
    return filtered_tokens

train_df['description_tokenized'] = train_df['description'].apply(lambda x:tokenize_text(x, stopwords.words('english')))
test_df['description_tokenized'] = test_df['description'].apply(lambda x:tokenize_text(x, stopwords.words('english')))
# создаем словарь

#словарь, составленный из описаний фильмов <word>:<id>
vocabulary = {}

max_val = 1000000

#добавляем зарезервированные слова
vocabulary["<PAD>"] = max_val + 2
vocabulary["<START>"] = max_val + 1
vocabulary["<UNKNOWN>"] = max_val

#посчитаем слова
for tokens in train_df.description_tokenized:
    for word in tokens:
        if word not in vocabulary.keys():
            vocabulary[word] = 1
        else:
            vocabulary[word] = vocabulary[word] + 1
            
#отсортируем слова по частоте
vocabulary = {k: v for k, v in sorted(vocabulary.items(), key=lambda item: item[1], reverse = True)}

#проиндексируем слова
cnt = 0
for k in vocabulary.keys():
    vocabulary[k] = cnt
    cnt = cnt + 1
#for
       
print('Количестов слов в словаре:',len(vocabulary))
print()
vocabulary
#создаем облегченный словарь для обучения
vocab_light = {}
for k, v in vocabulary.items():
    if v < 10000:
        vocab_light[k] = v

vocab_light
#описываем функции кодирования/декодирования слов

def encode_tokens (p_tokens, p_vocabulary):
    '''Кодирования токенов (слов) в индексы словаря
    
    :param p_tokens: список токенов
    :param p_vocabulary: словарь <word>:<id>, c обязательными значениями {<PAD>:0, <START>:1, <UNKNOWN>: 2}
    '''
    res = []
     
    res = [p_vocabulary.get(word, p_vocabulary['<UNKNOWN>']) for word in p_tokens]
    
    return [p_vocabulary['<START>']] + res

#encode_tokens

def dencode_tokens (p_encoded_tokens, p_vocabulary):
    '''Декодирование токенов: индексы словаря -> в тоекны (слова)
    
    :p_encoded_tokens: список индексов словаря
    :param p_vocabulary: словарь <word>:<id>, c обязательными значениями {<PAD>:0, <START>:1, <UNKNOWN>: 2}
    '''
    
    res = []
    
    for index in p_encoded_tokens: 
        for word, v_index in p_vocabulary.items():
            if index == v_index:
                res.append(word)
                break
            #if
    
    return res
    
#dencode_tokens

#!!! проверка
#d_id = 11
#print(train_df.description_tokenized[d_id])
#print()
#encoded_tokens = encode_tokens (train_df.description_tokenized[d_id], vocabulary)
#print (encoded_tokens)
#print()
#print (dencode_tokens (encoded_tokens, vocabulary))
#кодируем описание фильмов

train_df['description_encoded'] = train_df['description_tokenized'].apply(lambda x: encode_tokens (x, vocab_light))

train_df
test_df['description_encoded'] = test_df['description_tokenized'].apply(lambda x: encode_tokens (x, vocab_light))

test_df
#Готовим данные для обучения
train_data = train_df.description_encoded.to_numpy()
train_labels = pd.get_dummies(train_df['genre']).values

test_data = test_df.description_encoded.to_numpy()
test_labels = pd.get_dummies(test_df['genre']).values
# Посчитаем среднюю длинну описания, чтобы определить длинну последовательности
train_df['description_len'] = train_df['description_encoded'].apply (len)

print ('минимальная длина описания:', train_df.description_len.min())
print ('средняя длина описания:', round(train_df.description_len.mean()))
print ('максимальная длина описания:', train_df.description_len.max())

plt.hist(train_df.description_len, density = True)
# Приведем все цепочки в датасете к одной длине с помощью паддинга

MAX_SEQ_LEN = 70

train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data,
    value= vocabulary['<PAD>'],
    padding= 'post',
    maxlen= MAX_SEQ_LEN)

test_data = tf.keras.preprocessing.sequence.pad_sequences(
    test_data,
    value= vocabulary['<PAD>'],
    padding= 'post',
    maxlen= MAX_SEQ_LEN)

print('Тернировочные данные:')
print(train_data.shape)
print(train_data[0])
print()
print('Тестовые данные:')
print(test_data.shape)
print(test_data[0])
#Разбьем обучающий датасет на обучающий и валидационный

partial_x_train, x_val, partial_y_train, y_val = train_test_split(train_data, train_labels, test_size = 0.10, random_state = 42)

print(partial_x_train.shape, partial_y_train.shape)
print(x_val.shape, y_val.shape)
# Создадим рекурентную модель для классификации

VOCAB_SIZE = len(vocab_light)
# Размер векторного представления (эмбеддинга)
EMB_SIZE = 32
# Количество классов (жанров фильмов)
CLASS_NUM = y_val.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMB_SIZE),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(EMB_SIZE, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(EMB_SIZE, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)),
    tf.keras.layers.Dense(CLASS_NUM, activation= 'softmax'),
])

#model = tf.keras.Sequential([
#    tf.keras.layers.Embedding(VOCAB_SIZE, EMB_SIZE, input_length=train_data.shape[1]),
#    tf.keras.layers.SpatialDropout1D(0.2),
#    tf.keras.layers.LSTM(EMB_SIZE, dropout=0.2, recurrent_dropout=0.2),
#    tf.keras.layers.Dense(CLASS_NUM, activation= 'softmax'),
#])

model.summary()
#Обучение модели
BATCH_SIZE = 64
NUM_EPOCHS = 5

#Настраиваем объект для сохранения результатов работы модели
cpt_path = 'data/14_text_classifier.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(cpt_path, monitor='acc', verbose=1, save_best_only= True, mode='max')

model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

#запускаем обучение
history= model.fit(partial_x_train, partial_y_train, validation_data= (x_val, y_val), 
                   epochs= NUM_EPOCHS, batch_size= BATCH_SIZE, verbose= 1,
                   callbacks=[checkpoint])
#отобразим графики обучения

epochs = range(1, len(history.history['acc']) + 1)

plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.figure()
plt.plot(epochs, history.history['acc'], 'bo', label='Training acc')
plt.plot(epochs, history.history['val_acc'], 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
# оценим качество на тестовом датасете

results = model.evaluate(test_data, test_labels)

print('Test loss: {:.4f}'.format(results[0]))
print('Test accuracy: {:.2f} %'.format(results[1]*100))

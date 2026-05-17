import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import random
import sys
import os

# Создаём папки для результатов
os.makedirs('img', exist_ok=True)
os.makedirs('result', exist_ok=True)

# 1. Загружаем текстовый файл
with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(f"Загружено символов: {len(text)}")
print(f"Загружено слов: {len(text.split())}")

# 2. Выделяем алфавит (уникальные символы)
vocabulary = sorted(list(set(text)))
print(f"Размер алфавита: {len(vocabulary)}")

# 3. Создаём словари для кодирования символов
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))

# 4. Параметры
max_length = 40          # длина цепочки символов
step = 3                 # шаг смещения

sentences = []
next_chars = []

# 5. Формируем цепочки символов
for i in range(0, len(text) - max_length, step):
    sentences.append(text[i:i + max_length])
    next_chars.append(text[i + max_length])

print(f"Количество цепочек: {len(sentences)}")

# 6. Создаём тренировочные данные
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1

# 7. Строим LSTM модель
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Сохраняем структуру модели
from keras.utils import plot_model
plot_model(model, to_file='img/mod.png', show_shapes=True, show_layer_names=True)
print("Структура модели сохранена в img/mod.png")

# 8. Функция для выборки символа с температурой
def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 9. Функция для генерации текста
def generate_text(length, diversity):
    # Случайное начальное слово
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index:start_index + max_length]
    generated += sentence
    
    for i in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_indices[char]] = 1
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_char = indices_to_char[next_index]
        
        generated += next_char
        sentence = sentence[1:] + next_char
    
    return generated

# 10. Функция для отображения прогресса
def on_epoch_end(epoch, logs):
    if epoch % 10 == 0:
        print(f"\n--- Эпоха {epoch + 1} ---")
        print(generate_text(200, 0.2))
        print("---")

# 11. Callback для сохранения лучшей модели
checkpoint = ModelCheckpoint('model_checkpoint.h5', 
                             monitor='loss', 
                             save_best_only=True)

# 12. Обучение модели
print("\nНачало обучения...")
history = model.fit(X, y, 
                    batch_size=128, 
                    epochs=50,
                    callbacks=[checkpoint])

# 13. Сохраняем график обучения
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Потери')
plt.title('График обучения LSTM')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)
plt.savefig('img/learn.png')
print("График обучения сохранён в img/learn.png")

# 14. Генерируем текст
print("\nГенерация текста...")
generated_text = generate_text(1000, 0.2)

# 15. Сохраняем результат
with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)

print("Текст сохранён в result/gen.txt")
print(f"\nСгенерированный текст (первые 500 символов):\n{generated_text[:500]}")

# 16. Сохраняем пример сгенерированного текста в img/text.png
plt.figure(figsize=(12, 8))
plt.text(0.1, 0.5, generated_text[:1000], fontsize=10, wrap=True)
plt.axis('off')
plt.title('Сгенерированный текст LSTM')
plt.savefig('img/text.png')
print("Пример текста сохранён в img/text.png")

# 17. Выводим информацию о модели
print("\nИнформация о модели:")
model.summary()

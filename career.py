import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
model = Sequential()
#from M import sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.load(open('intents.json','r'))

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        # print(word_list)
        words.append(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print(words)
# print()
# print(documents)
words_temp = []
# print(words)
for word in words:
    for w in word:
        if w and w not in ignore_letters:
            # print(word)
            words_temp.append(lemmatizer.lemmatize(w))


# words_temp  = [lemmatizer.lemmatize(word) for word in words: if word not in ignore_letters]
words = sorted(set(words_temp))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training =[]
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_pattern = [lemmatizer.lemmatize(word.lower())for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
# print(len(train_x[0]))
# exit(0)
# for i in train_y:
#     if len(i) != 14:
#         print(len(i))
# print(len(train_x[0]),len(train_y))
# exit(0)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='CategoricalCrossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(x=np.array(train_x), y=np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel2.h5', save_format="h5")
print('Done')
print(model.input_shape)
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import pickle
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD  

if not os.path.exists('model'):
    os.makedirs('model')

lemmatizer = WordNetLemmatizer()

with open("intents.json", "r") as jsonFile:
    stringJsonFile = jsonFile.read()
    intents = json.loads(stringJsonFile)

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words2 = []
for word in words:
    if word not in ignore_letters:
        words2.append(lemmatizer.lemmatize(word))

words2 = sorted(set(words2))
classes = sorted(set(classes))

pickle.dump(words2, open('model/words2.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word) for word in word_patterns if word not in ignore_letters]

    for word in words2:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)

trainX = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(trainX), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('model/chatbot_model.keras')

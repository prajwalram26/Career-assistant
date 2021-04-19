import random
import json
import pickle
import numpy as np

import nltk
# nltk.download('all')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json','r'))

words = pickle.load(open('words.pkl', 'rb'))
#words = list(open("words.txt",'r', encoding='utf-8').readlines())hi
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel2.h5')

print(model.input_shape)

def clean_up_sentence(sentence):
    sentence_word = nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word

def bag_of_words(sentence):
    global words
    # print(words)
    sentance_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentance_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    # print("bow",len(bow))

    res = model.predict(np.array([bow]))[0]
    # print("llllll",len(res),res,"fff")
    ERROR_THRESHOLD = 0.25
    results = [[i, r]for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    # print("rss",results)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents':classes[r[0]], 'probability': str(r[1])})
    # print("rett",return_list)
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    result = "Error"
    # print("tag:",tag)
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    return result

print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    print(ints)
    res = get_response(ints, intents)
    print(res)
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
"""
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
"""

Lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    print(intent)
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        print(documents)
print(documents)

for w in words:
    # lemmatize each word to its base form
    w = Lemmatizer.lemmatize(w.lower())
    if w not in ignore_letters:
        words.append(w)
    
# remove duplicates
words = sorted(list(set(words)))



"""
my_bot = ChatBot(
    name='Bot',
    read_only=True,
    logic_adapters=["chatterbot.logic.BestMatch"] )

greetings = [
    'Hello',
    'Hi',
    'How are you?',
    'How do you do?', 
    'I am good,you?',
    'fine, you?',
    'alway fine, you?',
    'glad to hear that',
    'i feel good',
    'excellent,glad to hear that',
    'not so good',
    'sorry to hear that',
    'what is your name?',
    'my name is bot',
    'i m pybot,ask me anything',
]



q = ['pythagorean theorem',
          'a squared plus b squared equals c squared.']
math_talk_2 = ['law of cosines',
          'c**2 = a**2 + b**2 - 2 * a * b * cos(gamma)']
math_talk_3 = ['law of sines',
            'a = b * sin(gamma)']
list_trainer = ListTrainer(my_bot)

for item in (greetings, math_talk_1, math_talk_2):
          list_trainer.train(item)

print(my_bot.get_response("hi"))
print(my_bot.get_response("how are you?"))
print(my_bot.get_response("what is your name?"))
print(my_bot.get_response("i am good,you?"))
print(my_bot.get_response("show me the law of cosines"))
print(my_bot.get_response("show me the law of sines"))
print(my_bot.get_response("show me the law of pythagorean theorem"))


corpus_trainer = ChatterBotCorpusTrainer(my_bot)
corpus_trainer.train('chatterbot.corpus.english')

while True:
    try:
        user_input = input("You: ")
        bot_response = my_bot.get_response(user_input)
        print("Bot: ", bot_response)
    except(KeyboardInterrupt, EOFError, SystemExit):
        break;"""
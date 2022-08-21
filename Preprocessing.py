"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

import string
from spellchecker import SpellChecker
from TextScripts import notchangeWord

# Function to handle all the preprocessing
def Preprocessing(text,uncased=True,punct=False,checker=True):
    if uncased is True:
        text = cased_to_uncased(text)
    if punct is True:
        text = removePunctuation(text)
    if checker is True:
        text = removeMis(text)
    return text

# Function to lower case a string
def cased_to_uncased(text):
    if isinstance(text,str) is True:
        return text.lower()
    else:
        return text

# Function to remove punctuation
def removePunctuation(text):
    if isinstance(text,str) is True:
        text = text.replace("\n"," ")
        removerPunct = str.maketrans(string.punctuation," "*len(string.punctuation))
        text = text.translate(removerPunct)
        return text
    else:
        return text

# Function to handle small spelling mistake
def removeMis(text):
    spell = SpellChecker(language='fr',distance=1)
    spell.word_frequency.load_words(notchangeWord)
    new_text = []
    # Split the string into word
    for word in text.split():
        if any(char.isdigit() for char in word):
            new_text.append(word)
        else:
            new_text.append(spell.correction(word))
    new_text = " ".join(new_text)
    return new_text
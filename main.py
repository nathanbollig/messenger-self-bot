# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:44:53 2020

@author: NBOLLIG

Chat bot program that responds by cosine similarity matching with a data file or
by matching on a dictionary of questions and responses.

Takes in two input files:
    chatbot.txt - plaintext file
    responses.pickle - contains a python dictionary of messages as keys and responses as values

For a given user input, it checks all sentenses in chatbot.txt and keys in the 
response dictionary to find the best match based on cosine similarity. If the best
match is in the dictionary, it supplies the response. If the best match is in the 
plaintext file, it prints the matching sentence.

Similarity is measured by comparing TF-IDF vectors. The tokenizer considers
both unigram and bigram terms.

Some code snippets adapted from: 
https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e
"""

import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Read in chatbot.txt
f=open('chatbot.txt', 'r', errors = 'ignore')
raw=f.read()
raw=raw.lower()
sent_tokens = nltk.sent_tokenize(raw)

# Read in response dictionary
with open('responses.pickle', 'rb') as handle:
    responses = pickle.load(handle)        
num_of_responses = len(responses)
sent_tokens = list(responses.keys()) + sent_tokens
    
# Set uo tokenizer
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    output = []
    unigrams = LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    bigrams = list(nltk.bigrams(unigrams))
    output = unigrams
    for bigram in bigrams:
        unigrams.append(bigram[0] + ' ' + bigram[1])
    return output

# Prepare and tokenize stop words 
my_stop_words = ["what", "where", "when", "why", "how", "who"]
tokenized_stop_words = LemNormalize(' '.join(nltk.corpus.stopwords.words('english') + my_stop_words))

# Handle greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["greetings! except for a hot transistor, the weather's pretty good today!", "hidey-ho!", "*nods*", "hi there", "it's nice to meet you!"]
def greeting(sentence): 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Set up uncertain responses and goodbyes
UNCERTAIN_RESPONSES = ["I am sorry, I am a robot. I do not understand.", 
                       "My creator was not smart enough to teach me how to answer that.",
                       "I am just a robot, come on!",
                       "I really don't know what you are saying. Want to try another query?",
                       "I like your dialect! But I can't understand it :("]

GOODBYE_RESPONSES = ["Good bye!", "Bye bye!", "See you later, alligator!", "Hasta luego!"]

"""
Function to determine the chatbot's response to a user query.

Inputs:
    user_response - text input from the user
    threshold - if max cosine similarity is below threshold, then the response
        is randomly selected from the UNCERTAIN_RESPONSES list
"""
def response(user_response, threshold = 0.19):
    robo_response=''
    sent_tokens.append(user_response)    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=tokenized_stop_words)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    shift = int(np.abs(np.random.normal()))
    idx=vals.argsort()[0][-2-shift]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]    
    sim = vals[0][idx]
    if(req_tfidf==0 or np.abs(sim) < threshold):
        robo_response=robo_response+random.choice(UNCERTAIN_RESPONSES)
        return robo_response, sim
    else:
        if idx < num_of_responses:
            say = list(responses.values())[idx]
        else:
            say = sent_tokens[idx]
        
        robo_response = robo_response + say
        return robo_response, sim

# =============================================================================
# Main driver
# =============================================================================

if __name__ == 'main':
    flag=True
    print("NB: My name is Nathan. I will answer your queries. If you want to exit, type 'bye'!")
    while(flag == True):
        user_response = input()
        user_response=user_response.lower()
        
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you'):
                flag=False
                print("NB: You are welcome.")
            
            else:
                if(greeting(user_response)!=None):
                    print("NB: " + greeting(user_response))
                else:
                    print("NB: ", end="")
                    say, _ = response(user_response)
                    print(say)
                    sent_tokens.remove(user_response)
        
        else:
            flag=False
            print("NB: " + random.choice(GOODBYE_RESPONSES))
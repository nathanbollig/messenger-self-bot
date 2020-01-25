# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:21:00 2020

@author: NBOLLIG

Build a dictionary in which key is previous message and value is its response, based
on Facebook chat history. In order to qualify as a reponse, the message needs to
be sent within 12 hours of the previous message.

This script should be run in the same directory with a json Facebook Activity 
dump that includes messenger data.

The dictionary is pickled as "responses.pickle" in the working directory.

This has not been tested with any Facebook history dumps other than my own.

"""

import os
import json
import pickle

d = {}

"""
It appears that every recipient has their own directory containing exactly
one json file called "message_1.json", and that the messages are in reverse
chronological order. The following assumes this always holds.
"""

for entry in os.scandir("inbox"):
    path = os.path.join(entry.path, "message_1.json")
    with open(path) as json_file:
        data = json.load(json_file)
    
    for i in range(len(data['messages'])-2, -1, -1):
        prev_message = data['messages'][i+1]
        curr_message = data['messages'][i]
        prev_sender = prev_message['sender_name']
        curr_sender = curr_message['sender_name']
        prev_content = None
        curr_content = None
        if 'content' in prev_message:
            prev_content = prev_message['content']
        if 'content' in curr_message:
            curr_content = curr_message['content']
        if prev_sender != "Nathan Bollig" and curr_sender == "Nathan Bollig" and prev_content != None and curr_content != None:
            assert(curr_message['timestamp_ms'] > prev_message['timestamp_ms'])
            # message and response is stored in dictionary only if they are less than 12 hours apart
            if (curr_message['timestamp_ms'] - prev_message['timestamp_ms'])/1000.0 < 12*60:
                d[prev_content] = curr_content

with open('responses.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
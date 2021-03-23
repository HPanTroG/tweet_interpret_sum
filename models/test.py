from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, vstack
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import BertModel, AutoTokenizer
import re
import torch
import random
import os 
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
print(path)
sys.path.append(path)
from utils.tweet_preprocessing import tokenizeRawTweetText
import unicodedata as ud

def is_latin(word):
    return all(['LATIN' in ud.name(c) for c in word])

# tweet = "RT @IndianExpress: Google's ‘Person Finder’ tool is now working in Nepal's devastating earthquake of 2015: http://t.co/yemFNUsL5U http://t.…"
# print(tokenizeRawTweetText(tweet))
# prepro_text = tokenizeRawTweetText(tweet)
# for x in prepro_text.split(" "):
#     if is_latin(x):
#         print(x)
#     else:
#         print("not latin", x)

import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())

sent = "RT @IndianExpress: Google's ‘Person Finder’ tool is now working in Nepal's devastating earthquake of 2015: http://t.co/yemFNUsL5U http://t.…"
s = " ".join(w for w in nltk.wordpunct_tokenize(sent) \
         if w.lower() in words or not w.isalpha())
print(s)
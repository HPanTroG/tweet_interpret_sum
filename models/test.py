import os 
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
print(path)
sys.path.append(path)
from config.config import Config
import torch
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel, BertForSequenceClassification        

import pandas as pd
import transformers
print(transformers.__version__)

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
tweet = "<s> what is the simillary between saeed ajmal ban and pok flood in pakistan , they blamed india for both of reason xd"
print(tokenizer.tokenize(tweet))
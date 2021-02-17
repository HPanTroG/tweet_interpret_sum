
import time
import pandas as pd 
import numpy as np 
import torch 


def pad_sents(sents, pad_token):
    """
        :param sents: list[list[str]] list of tokenized sentences
        :param pad_token: int, pad token id
        @returns sents_padded: list[list[int]], list of tokenized sentences with padding shape(batch_size, max_sentence_length)
    """
    sents_padded = []
    max_len = max(len(s) for s in sents)
    for s in sents:
     
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        
        sents_padded.append(padded)
    return sents_padded

def sents_to_tensor(tokenizer, sents):
    """
        :param tokenizer
        :param sents: list[str], list of untokenized sentences
    """
    tokens_list = [tokenizer.tokenize(sent) for sent in sents]
   
    sents_lengths = [len(tokens) for tokens in tokens_list]
    sents_lengths = torch.tensor(sents_lengths)

    # pad sentences
    tokens_list_padded = pad_sents(tokens_list, '[PAD]')
    masks  = np.asarray(tokens_list_padded) != '[PAD]'
    masks_tensor = torch.tensor(masks, dtype= torch.long)
    tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    
    sents_tensor = torch.tensor(tokens_id_list, dtype = torch.long)
    return sents_tensor, masks_tensor, sents_lengths

def batch_iter(df, batch_size=60, shuffle = False):
    """
        split data into minibatch
    """
    
    if shuffle == True:
        df = df.reindex(np.random.permutation(df.index))
    
    length = len(df)

    for idx in range(0, length, batch_size):
        # dataframe can't split index label, should iter according index
        yield df.iloc[idx:min(idx+batch_size, length)]
       



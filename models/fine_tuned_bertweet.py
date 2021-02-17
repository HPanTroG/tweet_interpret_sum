import os 
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

from transformers import AutoTokenizer, BertForSequenceClassification
from utils.help_functions import sents_to_tensor
import torch
import numpy as np 
from torch.nn.utils.rnn import pack_padded_sequence 


class BertTweetClassification(torch.nn.Module):
    def __init__(self, num_class, bert_config='vinai/bertweet-base', device = 'cpu'):
        """
            :param num_class: number of classes to be classified
            :param bert_config: str, Bert model used
        """
        super(BertTweetClassification, self).__init__()
        self.num_class = num_class
        self.bert_config = bert_config
        self.model = BertForSequenceClassification.from_pretrained(self.bert_config, num_labels = self.num_class)
        self.device = device
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_config)

    def forward(self, sents):
        """
            :param sens: list[str], list of untokenized sentences
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents)
        output = self.model(input_ids = sents_tensor.to(self.device), attention_mask = masks_tensor.to(self.device))
        return output

  

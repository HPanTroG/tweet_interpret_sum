import numpy as np 
import pandas as pd 
from transformers import BertModel, AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import transformers
import torch.nn as nn
import torch
import random
import time
import re

class BertMTL(nn.Module):
    def __init__(self, bert_config='vinai/bertweet-base', exp_hidden_size=64, cls_hidden_size=64, n_cls_classes=2):
        super(BertMTL, self).__init__()
        self.base_bert = BertModel.from_pretrained(bert_config)
        self.exp_hidden_size = exp_hidden_size
        self.cls_hidden_size = cls_hidden_size
        self.cls_classes=n_cls_classes
        
        class ExpLayers(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(ExpLayers, self).__init__()
                self.exp_gru = nn.GRU(input_size, hidden_size)
                self.exp_linear = nn.Linear(hidden_size, 1, bias = True)
                self.exp_out = nn.Sigmoid()
            def forward(self, input):
                return self.exp_out(self.exp_linear(self.exp_gru(input)[0]))
        
        class ClsLayers(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(ClsLayers, self).__init__()
                self.cls_dropout = nn.Dropout(0.1)
                self.cls_linear1 = nn.Linear(input_size, hidden_size, bias = True)
                self.cls_tanh = nn.Tanh()
                self.cls_linear2 = nn.Linear(hidden_size, output_size, bias = True)
                self.cls_out = nn.Softmax(dim = -1)
            def forward(self, input):
                return self.cls_out(self.cls_linear2(self.cls_tanh(self.cls_linear1(self.cls_dropout(input)))))
        
        self.exp_layers = ExpLayers(self.base_bert.config.hidden_size, self.exp_hidden_size)
        self.cls_layers = ClsLayers(self.base_bert.config.hidden_size, self.cls_hidden_size, self.cls_classes)
    
    def forward(self, input_ids, input_masks):
        exp_output, cls_output = self.base_bert(input_ids, attention_mask=input_masks)
        exp_output = self.exp_layers(exp_output).squeeze() * input_masks
        cls_output = self.cls_layers(cls_output)
        return cls_output, exp_output

class BertMLTModel:
    def __init__(self, data, cls_labels, exp_labels, bert_config='vinai/bertweet-base', device = 'cpu', cls_hidden_size = 768, exp_hidden_size=768, random_state=12):
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_config)
        self.device = device 
        self.random_state = random_state
        
        self.data = np.array(['<s> '+x+ ' </s>' for x in data])
        self.cls_labels = torch.tensor(cls_labels, dtype=torch.long)
        self.n_cls_classes = len(set(cls_labels))
        self.tokenized_data, self.input_ids, self.attention_masks, self.tokenized_data_slides = self.tokenize_text(self.data)
        self.exp_labels_mapping = self.exp_labels_mapping(self.data, exp_labels)
        self.tokenized_data, self.input_ids = np.array(self.tokenized_data), torch.tensor(self.input_ids, dtype = torch.long)
        self.attention_masks, self.tokenized_data_slides= torch.tensor(self.attention_masks, dtype = torch.long), np.array(self.tokenized_data_slides)
        self.exp_labels_mapping = torch.tensor(self.exp_labels_mapping, dtype = torch.long)
        self.model = BertMTL(bert_config = bert_config, cls_hidden_size = cls_hidden_size, exp_hidden_size = exp_hidden_size, n_cls_classes = self.n_cls_classes)
        self.model.to(device)

    

    def pad_sents(self, sents, pad_token):
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
    
    def text_to_exp_labels(self, org_text, explan_text):
        """
            param org_text: str, original text
            param explan_text: list[str], explanation snippets
            @return list[int], list of 0/1 values to label whether each token in org_text is a part of explan_text 
        """
        labels = org_text
        for chunk in explan_text:
            labels = re.sub(chunk, '1 '*len(chunk.split( )), labels)

        labels = re.sub('[^1 ]', '0', labels)
        labels = re.sub('  ', ' ', labels).strip().split(" ")
        labels = [1 if i == '1' else 0 for i in labels]
        return labels
    def exp_labels_mapping(self, sents, exp_labels):
        """
            :param sents: list[str], list of input sentences
            :param exp_labels: list[list[str]], list of explanation snippets of input sentences
            @return exp_labels_mapping: list[int], list of 0/1 to specify whether the token is a part of explanation
            
        """
        tokenized_sents = [' '.join(self.tokenizer.tokenize(sent)) for sent in sents]
       
        tokenized_exps = [[' '.join(self.tokenizer.tokenize(exp))] for exps in exp_labels for exp in exps]
        exp_labels = [self.text_to_exp_labels(sent, exp) for sent, exp in zip(tokenized_sents, tokenized_exps)]
        # padding 0
        max_length = max([len(sent.split(" ")) for sent in tokenized_sents])
        exp_labels = [label+[0]*(max_length-len(label)) for label in exp_labels]
        return exp_labels

    def tokenize_text(self, sents, padding_token = '<pad>'):
        """
            :param sents: list[str], list of untokenized sentences
            @return: tokenized_sents: list[int], list of token ids  
        """
        tokens_list = []
        tokens_spans = []
        for sent in sents:
            tokens = []
            spans = []
            start_w = 0
            for w in sent.split(" "):
                tokens.extend(self.tokenizer.tokenize(w))
                end_w = len(tokens)
                spans.append((start_w, end_w))
                start_w = end_w
            tokens_list.append(tokens)
            tokens_spans.append(spans)
        # pad sentences
        tokens_list_padded = self.pad_sents(tokens_list, padding_token)
        attention_masks  = np.asarray(tokens_list_padded) != padding_token
        tokens_id_list_padded =[self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
        return tokens_list, tokens_id_list_padded, attention_masks, tokens_spans
    
    def predict(self, X_test, batch_size = 128):
        self.model.eval()
        y_cls_preds = []
        y_exp_preds = []
        sents_tensor, masks_tensor, sents_lengths = self.convert_sents_to_ids_tensor(X_batch)
        with torch.no_grad():
            for batch_start in range(0, len(X_test), batch_size):
                batch_end = min(batch_start+batch_size, len(X_test))
                X_batch = X_test[batch_start: batch_end]
                
                output = self.model(input_ids = sents_tensor.to(self.device), attention_mask = masks_tensor.to(self.device))[0].cpu()
                
                outputs+= np.argmax(output, 1).flatten()
        return outputs
        y_cls_pred, y_exp_pred = self.model(X_test)

    def resampling_rebalanced_crossentropy(self, seq_reduction = 'none'):
        def loss(y_pred, y_true):
            prior_pos = torch.mean(y_true, dim=-1, keepdims=True)
            prior_neg = torch.mean(1-y_true, dim=-1, keepdim=True)
            eps=1e-10
            weight = y_true / (prior_pos + eps) + (1 - y_true) / (prior_neg + eps)
            ret =  -weight * (y_true * (torch.log(y_pred + eps)) + (1 - y_true) * (torch.log(1 - y_pred + eps)))
            if seq_reduction == 'mean':
                return torch.mean(ret, dim=-1)
            elif seq_reduction == 'none':
                return ret
        return loss

    def train(self, test_size = 0.15, n_epochs =10, cls_class_weights = None, exp_class_weights = None, train_batch_size = 64, test_batch_size = 128, par_lambda=0.5):
        self.model.train()
        data_indices = [i for i in range(len(self.data))]
        train_indices, test_indices = train_test_split(data_indices, random_state = self.random_state, stratify = self.cls_labels)

        train_data, test_data = self.data[train_indices], self.data[test_indices]
        train_tokenized_data, test_tokenized_data = self.tokenized_data[train_indices], self.tokenized_data[test_indices]
        train_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[test_indices]
        train_attention_masks, test_attention_masks = self.attention_masks[train_indices], self.attention_masks[test_indices]
        train_tokenized_data_slides, test_tokenized_data_slides = self.tokenized_data_slides[train_indices], self.tokenized_data_slides[test_indices]
        train_exp_labels, test_exp_labels = self.exp_labels_mapping[train_indices], self.exp_labels_mapping[test_indices]
        train_cls_labels, test_cls_labels = self.cls_labels[train_indices], self.cls_labels[test_indices]


        n_batches = int(np.ceil(len(train_indices)/train_batch_size))
        print("#batches for each epoch: {}".format(n_batches))
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_step = n_batches * n_epochs

        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=total_step)
        if cls_class_weights!=None: 
            cls_class_weights = torch.tensor([cls_class_weights[k] for k in sorted(cls_class_weights.keys())])
            cls_criterion = torch.nn.CrossEntropyLoss(weight = cls_class_weights.to(self.device), reduction = 'none')
        else:
            cls_criterion =  torch.nn.CrossEntropyLoss(reduction = 'none')
        
        exp_criterion = self.resampling_rebalanced_crossentropy(seq_reduction = 'none')
        for epoch in range(n_epochs):
            print("...............................")
            total_train_loss = 0
            begin_time = time.time()
            epoch_indices = random.sample([i for i in range(len(train_input_ids))], len(train_input_ids))

            for batch_start in range(0, len(train_input_ids), train_batch_size):
                batch_end = min(batch_start+train_batch_size, len(train_input_ids))
                batch_input_ids = train_input_ids[batch_start: batch_end]
                batch_attention_masks = train_attention_masks[batch_start: batch_end]
                batch_cls_labels = train_cls_labels[batch_start: batch_end]
                batch_exp_labels = train_exp_labels[batch_start: batch_end]

                cls_preds, exp_preds = self.model(batch_input_ids.to(self.device), batch_attention_masks.to(self.device))
            

                cls_loss = cls_criterion(cls_preds, batch_cls_labels.to(self.device)).mean(dim = -1).sum()
                exp_loss = exp_criterion(exp_preds, batch_exp_labels.to(self.device).float()).mean(dim = -1).sum()
                loss = cls_loss + par_lambda * exp_loss
                optimizer.zero_grad()
                total_train_loss += loss.item()
                loss.backward()
                # clip the norm of the gradients to 1, to prevent exploding
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # update parameters
                optimizer.step()

                # update learning rate
                scheduler.step()
            print("Epoch: {}, training loss: {}, time: {}".format(epoch, total_train_loss/n_batches, time.time()-begin_time))


        cls_pred_labels, exp_pred_labels = self.predict(test_input_ids, test_attention_masks, test_batch_size)
        # print(len(cls_pred_labels), len(test_cls_labels), len(test_input_ids), len(exp_pred_labels))
        
        exp_true = self.max_pooling(test_exp_labels, test_tokenized_data_slides, test_data)
        # for i in range(len(exp_true)):
        #     print(test_cls_labels[i])
        #     print(exp_true[i])
        #     print(test_exp_labels[i])
        #     input()
        exp_pred = self.max_pooling(exp_pred_labels, test_tokenized_data_slides, test_data)

        i = 0
        with open("result_exp.out", "w") as f:
            for data, cls_true, cls_pred, exp_t, exp_p in zip(test_data, test_cls_labels, cls_pred_labels, exp_true, exp_pred):
                f.write(data +"\n")
                f.write("cls_true+predicted: {}-{}\n".format(cls_true, cls_pred))
                f.write("exp_true: {}\n".format(exp_t))
                f.write("exp_pred: {}\n".format(exp_p))
                f.write("..................\n")

        print("Prediction task: ")
        print(classification_report(test_cls_labels, cls_pred_labels))
        exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
        print("Explanation task:", exp_f1)
        


    def max_pooling(self, y_values, data_slides, data):
        pooled_values = []

        for y, y_slides, data in zip(y_values, data_slides, data):
            pooled_y = []
            for tup in y_slides:

                pooled_y.append(int(max(y[tup[0]:tup[1]])))
            pooled_values.append(pooled_y)
        
        return pooled_values

    def predict(self, input_ids, attention_masks, test_batch_size):
        self.model.eval()

        cls_preds = []
        exp_preds = []
        batch_start = 0

        with torch.no_grad():
            for batch_start in range(0, len(input_ids), test_batch_size):
                batch_end = min(batch_start+test_batch_size, len(input_ids))
                batch_input_ids = input_ids[batch_start: batch_end]
                batch_attention_masks = attention_masks[batch_start: batch_end]
       
                cls_outs, exp_outs = self.model(batch_input_ids.to(self.device), batch_attention_masks.to(self.device))
                
                cls_outs = cls_outs.argmax(dim = -1).cpu()
                exp_outs = torch.round(exp_outs).long().cpu()
                
                cls_preds += cls_outs
                exp_preds += exp_outs




        return cls_preds, exp_preds


                




        


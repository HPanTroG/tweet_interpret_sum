import numpy as np 
import pandas as pd 
from transformers import BertModel, AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import transformers
import torch.nn as nn
import torch
import random
import time
import re
import sys

class BertMTLBase(nn.Module):
    def __init__(self, bert_config='vinai/bertweet-base', exp_hidden_size=64, cls_hidden_size=64, n_cls_classes=2):
        super(BertMTLBase, self).__init__()
        self.bert_config = bert_config
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
    
        exp_output, cls_output = self.base_bert(input_ids = input_ids, attention_mask=input_masks)
        exp_output = self.exp_layers(exp_output).squeeze() * input_masks
        cls_output = self.cls_layers(cls_output)
        return cls_output, exp_output
    
    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BertMTLBase(**args)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(bert_config=self.bert_config, cls_hidden_size= self.cls_hidden_size, exp_hidden_size = self.exp_hidden_size, n_cls_classes = self.cls_classes),
            'state_dict': self.state_dict(),
          
        }
        torch.save(params, path)

class BertMLTModel:
    def __init__(self, data, cls_labels, exp_labels, bert_config='vinai/bertweet-base', device = 'cpu', cls_hidden_size = 768, exp_hidden_size=768, random_state=12, n_cls_classes=0):
        
        self.bert_config = bert_config
        self.device = device
        self.cls_hidden_size = cls_hidden_size 
        self.exp_hidden_size = exp_hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(bert_config)
        self.device = device 
        self.random_state = random_state
        if cls_labels!=None:
            self.data = np.array(['<s> '+x+ ' </s>' for x in data])
            self.cls_labels = torch.tensor(cls_labels, dtype=torch.long)
            self.exp = exp_labels
            self.n_cls_classes = len(set(cls_labels))
            self.tokenized_data, self.input_ids, self.attention_masks, self.tokenized_data_slides = self.tokenize_text(self.data)
            self.exp_labels_mapping = self.map_exp_labels(self.data, exp_labels)
            self.tokenized_data, self.input_ids = np.array(self.tokenized_data), torch.tensor(self.input_ids, dtype = torch.long)
            self.attention_masks, self.tokenized_data_slides= torch.tensor(self.attention_masks, dtype = torch.long), np.array(self.tokenized_data_slides)
            self.exp_labels_mapping = torch.tensor(self.exp_labels_mapping, dtype = torch.long)
        else: 
            self.n_cls_classes = n_cls_classes
        self.model = BertMTLBase(bert_config = self.bert_config, cls_hidden_size = self.cls_hidden_size, exp_hidden_size = self.exp_hidden_size, n_cls_classes = self.n_cls_classes)
    

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
        try:
            for chunk in explan_text:
               labels = re.sub(re.escape(chunk), '1 '*len(chunk.split( )), labels)
            labels = re.sub('[^1 ]', '0', labels)
            labels = re.sub('  ', ' ', labels).strip().split(" ")
            labels = [1 if i == '1' else 0 for i in labels]
        except Exception as e:
            print("EXCEPTION.............................. ")
            print(org_text)
            print(explan_text)
            
        return labels

    def map_exp_labels(self, sents, exp_labels):
        """
            :param sents: list[str], list of input sentences
            :param exp_labels: list[list[str]], list of explanation snippets of input sentences
            @return exp_labels_mapping: list[int], list of 0/1 to specify whether the token is a part of explanation
            
        """
        tokenized_sents = [' '.join(self.tokenizer.tokenize(sent)) for sent in sents]
       
        tokenized_exps = [[' '.join(self.tokenizer.tokenize(exp)) for exp in exps] for exps in exp_labels]
        exp_labels = [self.text_to_exp_labels(sent, exp) for sent, exp in zip(tokenized_sents, tokenized_exps)]
        # padding 0
        max_length = max([len(sent.split(" ")) for sent in tokenized_sents])
        exp_labels = [label+[0]*(max_length-len(label)) for label in exp_labels]
        return exp_labels

    def tokenize_text(self, sents, padding_token = '<pad>'):
        """
            :param sents: list[str], list of untokenized sentences
            @return: tokenized_lists: list[str], list of tokenized tokens
            @return: tokens_id_list_padded: list[int], list of ids of tokenized tokens
            @return: tokens_spans, list[(int, int)], list of start_span, end_span to recover original unknown words
                      
        """
        tokens_list = []
        tokens_spans = []
        for sent in sents:
            tokens = []
            spans = []
            start_w = 0
            for w in sent.split(" "):
                tokenized = self.tokenizer.tokenize(w)
                if start_w+len(tokenized)>= 130:
                    tokens.extend(self.tokenizer.tokenize("</s>"))
                    end_w = len(tokens)
                    spans.append((start_w, end_w))
                    break
                tokens.extend(tokenized)
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

    def fit(self, train_input_ids, train_attention_masks, train_cls_labels, train_exp_labels, cls_criterion, exp_criterion, exp_weight, train_batch_size=64):
        
        total_train_loss = 0
        n_batches = int(np.ceil(len(train_input_ids)/train_batch_size))
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
            loss = cls_loss + exp_weight * exp_loss
            self.optimizer.zero_grad()
            total_train_loss += loss.item()
            
            loss.backward()
            # clip the norm of the gradients to 1, to prevent exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # update learning rate
            self.scheduler.step()
        return total_train_loss/n_batches

    def cross_validate(self, n_folds = 5, test_size = 0.3, best_model_path = "", train_batch_size = 64, n_epochs=20, test_batch_size = 128,  cls_weights = None, out = "", patience=3, idx_label_map={}, exp_weights=[0.01], best_epoch_temp=""):
        
        kfold = StratifiedShuffleSplit(n_splits = n_folds, test_size =test_size, random_state=self.random_state)
        fold = 0
        with open(out, "w") as f:
            f.write("............................\n")
        if cls_weights!=None: 
            cls_weights = torch.tensor([cls_weights[k] for k in sorted(cls_weights.keys())])
            cls_criterion = torch.nn.CrossEntropyLoss(weight = cls_weights.to(self.device), reduction = 'none')
        else:
            cls_criterion =  torch.nn.CrossEntropyLoss(reduction = 'none')
            
        # exp_criterion =  torch.nn.BCEWithLogitsLoss(reduction = 'none')
        exp_criterion = self.resampling_rebalanced_crossentropy(seq_reduction = 'none')
        print("Original data: ", len(self.data))
        for train_indices, remaining_indices in kfold.split(self.data, self.cls_labels):
            valid_indices, test_indices= train_test_split(remaining_indices, test_size =0.5, random_state=self.random_state, stratify=self.cls_labels[remaining_indices])
            print("Fold: {}........................................".format(fold))
            fold+=1

            train_data, valid_data, test_data = self.data[train_indices], self.data[valid_indices], self.data[test_indices]
            train_tokenized_data, valid_tokenized_data, test_tokenized_data = self.tokenized_data[train_indices], self.tokenized_data[valid_indices], self.tokenized_data[test_indices]
            train_input_ids, valid_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[valid_indices], self.input_ids[test_indices]
            train_attention_masks, valid_attention_masks, test_attention_masks = self.attention_masks[train_indices], self.attention_masks[valid_indices], self.attention_masks[test_indices]
            train_tokenized_data_slides, valid_tokenized_data_slides, test_tokenized_data_slides = self.tokenized_data_slides[train_indices], self.tokenized_data_slides[valid_indices], self.tokenized_data_slides[test_indices]
            train_exp_labels, valid_exp_labels, test_exp_labels = self.exp_labels_mapping[train_indices], self.exp_labels_mapping[valid_indices], self.exp_labels_mapping[test_indices]
            train_cls_labels, valid_cls_labels, test_cls_labels = self.cls_labels[train_indices], self.cls_labels[valid_indices], self.cls_labels[test_indices]

          
            best_exp_weight = -1
            best_f1 = 0.0
            for exp_weight in exp_weights:
                print("...........Exp weight: {}...................".format(exp_weight))
                self.model = BertMTLBase(bert_config = self.bert_config, cls_hidden_size = self.cls_hidden_size, exp_hidden_size = self.exp_hidden_size, n_cls_classes = self.n_cls_classes)
                self.model.to(self.device)
                self.model.train()
                n_batches = int(np.ceil(len(train_indices)/train_batch_size))
                print("#batches for each epoch: {}".format(n_batches))
                self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
                total_step = n_batches * n_epochs

                self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                        num_warmup_steps=0,
                                                                        num_training_steps=total_step)
            
                valid_loss_hist = []
                best_epoch = 0
                
                for epoch in range(n_epochs):
                    begin_time = time.time()
                    train_loss = self.fit(train_input_ids, train_attention_masks, train_cls_labels, train_exp_labels, cls_criterion, exp_criterion, exp_weight, train_batch_size)
                    print("Epoch: {}, train_loss: {}, time: {}".format(epoch, train_loss, time.time()-begin_time))                

                #evaluate on validation set
                cls_pred_labels, exp_pred_labels, cls_pred_probs = self.predict(valid_input_ids, valid_attention_masks, test_batch_size)
                exp_true = self.max_pooling(valid_exp_labels, valid_tokenized_data_slides, valid_data)
                exp_pred = self.max_pooling(exp_pred_labels, valid_tokenized_data_slides, valid_data)

                exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
                exp_precision = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
                exp_recall = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
                cls_f1 = f1_score(valid_cls_labels, cls_pred_labels, average='macro')
                f1 = (exp_f1 + cls_f1)/2

                with open(out, "a") as f:
                    f.write("Fold: {}, exp_weight: {}, valid_exp_f1: {}, valid_cls_f1: {}, f1_mean: {}, valid_exp_P: {}, valid_exp_R:{}\n".format(fold, exp_weight, 
                                    exp_f1, cls_f1, f1, exp_precision, exp_recall))
                if f1> best_f1:
                    best_f1 = f1
                    best_exp_weight = exp_weight
                    print("++ Save model with best exp...: ", best_exp_weight)
                    self.model.save(best_model_path)

            # evaluation on test set
            params = torch.load(best_model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(params['state_dict'])
            self.model.to(self.device)
            cls_pred_labels, exp_pred_labels, cls_pred_probs = self.predict(test_input_ids, test_attention_masks, test_batch_size)
            exp_true = self.max_pooling(test_exp_labels, test_tokenized_data_slides, test_data)
            exp_pred = self.max_pooling(exp_pred_labels, test_tokenized_data_slides, test_data)

            exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
            exp_precision = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
            exp_recall = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])

            cls_f1 = f1_score(test_cls_labels, cls_pred_labels, average='macro')
            print("........................................................")
            print("Test Evaluation:")
            print(classification_report(test_cls_labels, cls_pred_labels, target_names =  [idx_label_map[key] for key in range(len(idx_label_map))]))
            print("Explanation Token F1:", exp_f1)

            with open(out, 'a') as f:
                f.write("Fold: {} ++++Test Evaluation+++++\n".format(fold))
                f.write("Best exp weight: {}".format(best_exp_weight))
                f.write(classification_report(test_cls_labels, cls_pred_labels, target_names =  [idx_label_map[key] for key in range(len(idx_label_map))]))
                f.write("\nexp_P: {}, exp_R: {}, exp_f1: {}, cls_f1: {}, f1_mean: {}".format(exp_precision, exp_recall, exp_f1, cls_f1, (exp_f1+cls_f1)/2))
                f.write("=================================================================\n")
            if n_folds ==1:
                with open(out, "a") as f:
                    for data, cls_true, cls_pred, cls_prob, exp_t, exp_p in zip(test_data, test_cls_labels, cls_pred_labels, cls_pred_probs, exp_true, exp_pred):
                        f.write(data +"\n")
                        text = data.split(' ')
                        f.write("cls_true+predicted: {}-{}, prob: {}\n".format(cls_true, cls_pred, cls_prob))
                        f.write("exp_true: {}, {}\n".format(exp_t, ' '.join(text[i] for i in range(len(text)) if exp_t[i]==1)))
                        f.write("exp_pred: {}, {}\n".format(exp_p, ' '.join(text[i] for i in range(len(text)) if exp_p[i]==1)))
                        f.write("..................\n")

                

    def train(self, test_size = 0.15, n_epochs =10, cls_weights = None, exp_weight=0.01, train_batch_size = 64, test_batch_size = 128, idx_label_map = {}, model_path=""):
        self.model = BertMTLBase(bert_config = self.bert_config, cls_hidden_size = self.cls_hidden_size, exp_hidden_size = self.exp_hidden_size, n_cls_classes = self.n_cls_classes)
        self.model.to(self.device)
        self.model.train()
        data_indices = [i for i in range(len(self.data))]
        if test_size !=0:
            train_indices, test_indices = train_test_split(data_indices, test_size = test_size, random_state = self.random_state, stratify = self.cls_labels)
            print(test_indices)
        else:
            random.shuffle(data_indices)
            train_indices, test_indices = data_indices, data_indices
        
        print("Train size: {}, test size: {}".format(len(train_indices), len(test_indices)))

        train_data, test_data = self.data[train_indices], self.data[test_indices]
        train_tokenized_data, test_tokenized_data = self.tokenized_data[train_indices], self.tokenized_data[test_indices]
        train_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[test_indices]
        train_attention_masks, test_attention_masks = self.attention_masks[train_indices], self.attention_masks[test_indices]
        train_tokenized_data_slides, test_tokenized_data_slides = self.tokenized_data_slides[train_indices], self.tokenized_data_slides[test_indices]
        train_exp_labels, test_exp_labels = self.exp_labels_mapping[train_indices], self.exp_labels_mapping[test_indices]
        train_cls_labels, test_cls_labels = self.cls_labels[train_indices], self.cls_labels[test_indices]
        
       
        n_batches = int(np.ceil(len(train_indices)/train_batch_size))
        print("#batches for each epoch: {}".format(n_batches))
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_step = n_batches * n_epochs

        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=total_step)
        if cls_weights!=None: 
            cls_weights = torch.tensor([cls_weights[k] for k in sorted(cls_weights.keys())])
            cls_criterion = torch.nn.CrossEntropyLoss(weight = cls_weights.to(self.device), reduction = 'none')
        else:
            cls_criterion =  torch.nn.CrossEntropyLoss(reduction = 'none')
            
        exp_criterion = self.resampling_rebalanced_crossentropy(seq_reduction = 'none')
        for epoch in range(n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_input_ids, train_attention_masks, train_cls_labels, train_exp_labels, cls_criterion, exp_criterion, exp_weight, train_batch_size)
            print("Epoch: {}, training loss: {}, time: {}".format(epoch, train_loss, time.time()-begin_time))
        if model_path!="":
            self.model.save(model_path)

        cls_pred_labels, exp_pred_labels, cls_pred_probs = self.predict(test_input_ids, test_attention_masks, test_batch_size)
        exp_true = self.max_pooling(test_exp_labels, test_tokenized_data_slides, test_data)
        exp_pred = self.max_pooling(exp_pred_labels, test_tokenized_data_slides, test_data)

        i = 0
        # with open("result_exp.out", "w") as f:
        #     for data, cls_true, cls_pred, exp_t, exp_p in zip(test_data, test_cls_labels, cls_pred_labels, exp_true, exp_pred):
        #         f.write(data +"\n")
        #         text = data.split(' ')
        #         f.write("cls_true+predicted: {}-{}\n".format(cls_true, cls_pred))
        #         f.write("exp_true: {}, {}\n".format(exp_t, ' '.join(text[i] for i in range(len(text)) if exp_t[i]==1)))
        #         f.write("exp_pred: {}, {}\n".format(exp_p, ' '.join(text[i] for i in range(len(text)) if exp_p[i]==1)))
                
        #         f.write("..................\n")

        print("Prediction task: ")
        print(classification_report(test_cls_labels, cls_pred_labels, target_names =  [idx_label_map[key] for key in range(len(idx_label_map))]))
        exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
        print("Explanation task token-f1:", exp_f1)
        


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

        cls_labels = []
        exp_preds = []
        cls_probs = []
        batch_start = 0
    
        with torch.no_grad():
            for batch_start in range(0, len(input_ids), test_batch_size):
                batch_end = min(batch_start+test_batch_size, len(input_ids))
                batch_input_ids = input_ids[batch_start: batch_end]
                batch_attention_masks = attention_masks[batch_start: batch_end]
                print("Batch... input: {}, attention: {}\n".format(batch_input_ids.shape, batch_attention_masks.shape))
                cls_outs, exp_outs = self.model(batch_input_ids.to(self.device), batch_attention_masks.to(self.device))
                
                cls_outs = cls_outs.max(dim = -1)
                cls_pred_labels = cls_outs.indices.cpu()
                cls_pred_probs = cls_outs.values.cpu()
                exp_outs = torch.round(exp_outs).long().cpu()
                
                cls_labels += cls_pred_labels
                exp_preds += exp_outs
                cls_probs += cls_pred_probs

        return cls_labels, exp_preds, cls_probs
    
    def check_explanation(self, saved_model="", idx_label_map={}, batch_size=128): 
        if saved_model == "":
            print("Please enter the path of saved model")
            return
        
        params = torch.load(saved_model, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(params['state_dict'])
        self.model.to(self.device)
        exp_true = self.max_pooling(self.exp_labels_mapping, self.tokenized_data_slides, self.data)
        cls_labels = list(self.cls_labels.numpy())
        exp_true_dis = {}
        for i, exp in enumerate(exp_true):
            exp_true_dis[idx_label_map[cls_labels[i]]] = exp_true_dis.get(idx_label_map[cls_labels[i]], [])+[sum(exp)/(len(exp)-2)]

        exp_true_dis = {key:np.mean(value) for key, value in exp_true_dis.items()}
        print("True rationale distribution: ", exp_true_dis)

        cls_pred_labels, exp_pred_labels, cls_pred_prob = self.predict(self.input_ids, self.attention_masks, self.cls_labels, self.exp_labels_mapping, batch_size)
        exp_pred = self.max_pooling(exp_pred_labels, self.tokenized_data_slides, self.data)
        print("F1: ", f1_score(self.cls_labels, cls_pred_labels, average = 'macro'))
        cls_pred_labels = [int(i) for i in cls_pred_labels]        
        exp_pred_dis_cls_true = {}
        exp_pred_dis_cls_pred = {}
        for i, exp in enumerate(exp_pred):
            exp_pred_dis_cls_pred[idx_label_map[cls_pred_labels[i]]] = exp_pred_dis_cls_pred.get(idx_label_map[cls_pred_labels[i]], [])+[sum(exp)/(len(exp)-2)]
            exp_pred_dis_cls_true[idx_label_map[cls_labels[i]]] = exp_pred_dis_cls_true.get(idx_label_map[cls_labels[i]], [])  +[sum(exp)/(len(exp)-2)]
        exp_pred_dis_cls_pred = {key:np.mean(value) for key, value in exp_pred_dis_cls_pred.items()}
        exp_pred_dis_cls_true = {key:np.mean(value) for key, value in exp_pred_dis_cls_true.items()}
        
        print("Pred rationale distribution (based on pred cls labels)", exp_pred_dis_cls_pred)
        print("Pred rationale distribution (based on true cls labels)", exp_pred_dis_cls_true)

    def classify_new_data(self, model_path="", input={}, output_path = "", batch_size = 128, idx_label_map={}):

        if model_path == "":
            print("PLEASE ENTER MODEL PATH......")
            return
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(params['state_dict'])
        self.model.to(self.device)
        with open(output_path, "w") as f:
            f.write("tweet_id\tcreated_at\ttweet_text\tpredicted_labels\texplanation\n")
        prepro_text = np.array(['<s> '+x+ ' </s>' for x in input['prepro_text']])
        # extract input_ids, attention_masks
        tokenized_data, input_ids, attention_masks, tokenized_data_slides = self.tokenize_text(prepro_text)

        tokenized_data, input_ids = np.array(tokenized_data), torch.tensor(input_ids, dtype = torch.long)
        attention_masks, tokenized_data_slides= torch.tensor(attention_masks, dtype = torch.long), np.array(tokenized_data_slides)
        
        cls_pred, exp_pred_labels, cls_pred_prob = self.predict(input_ids, attention_masks, batch_size)

        exp_pred = self.max_pooling(exp_pred_labels, tokenized_data_slides, prepro_text)

        i = 0
        with open(output_path, "a") as f:
            for id, created_at, txt, prepro_txt, cls_label, cls_prob, exp_label  in zip(input['id'], input['created_at'], input['text'], prepro_text, cls_pred, cls_pred_prob, exp_pred):
                text = prepro_txt.split(" ")
                f.write("{}\t{}\t{}\t".format(id, created_at, txt))
                pred_exp_text = ' '.join(text[i] for i in range(len(exp_label)) if exp_label[i]==1)
                f.write("{}\t{}\t{}\n".format(idx_label_map[int(cls_label)], cls_prob, pred_exp_text))
               
        


                




        


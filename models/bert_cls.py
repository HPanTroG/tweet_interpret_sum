import torch
import numpy as np
import pandas as pd
from sklearn.metrics import  f1_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from collections import Counter
import time, datetime
import transformers
import warnings
import random
import shutil
warnings.simplefilter('ignore')


class Bertweet:
    def __init__(self, data = None, labels =None, train_data = None, train_labels = None, 
                test_data=None, test_labels = None,bert_config='vinai/bertweet-base', device = 'cpu', random_state=12):

        self.data = np.array(["<s> "+x + " </s>" for x in data]) # add starting/ending tokens
        self.labels = np.array(labels)
        self.num_classes = len(set(labels))
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(bert_config)
        self.random_state = random_state
        self.bert_config = bert_config
        self.model = BertForSequenceClassification.from_pretrained(self.bert_config, num_labels = self.num_classes)
       


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

    def convert_sents_to_ids_tensor(self, sents):
        """
            :param tokenizer
            :param sents: list[str], list of untokenized sentences
        """
        tokens_list = [self.tokenizer.tokenize(sent) for sent in sents]
    
        sents_lengths = [len(tokens) for tokens in tokens_list]
        sents_lengths = torch.tensor(sents_lengths)

        # pad sentences
        tokens_list_padded = self.pad_sents(tokens_list, '<pad>')
        masks  = np.asarray(tokens_list_padded) != '<pad>'
        masks_tensor = torch.tensor(masks, dtype= torch.long)
        tokens_id_list = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
        
        sents_tensor = torch.tensor(tokens_id_list, dtype = torch.long)
        return sents_tensor, masks_tensor, sents_lengths
    

    def predict(self, data, labels, loss_func, batch_size=128, val = False):
        self.model.eval()
        n_batches = int(np.ceil(len(data)/batch_size)) 
        total_loss = 0
        outputs = []

        with torch.no_grad():
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start+batch_size, len(data))
                X_batch = data[batch_start: batch_end]
                y_batch = labels[batch_start: batch_end]
                sents_tensor, masks_tensor, sents_lengths = self.convert_sents_to_ids_tensor(X_batch)
                output = self.model(input_ids = sents_tensor.to(self.device), attention_mask = masks_tensor.to(self.device))[0]
                loss = loss_func(output, torch.tensor(y_batch, dtype = torch.long).to(self.device))
                total_loss += loss.item()
                outputs+= np.argmax(output.cpu(), 1).flatten()
                
        loss = total_loss/n_batches
        return loss, outputs
    

    def fit(self, data, labels, train_batch_size, loss_func):
        train_indices = [i for i in range(len(data))]
        ys = []
        train_loss = 0
        n_batches = int(np.ceil(len(data)/train_batch_size))
        epoch_indices = random.sample(train_indices, len(train_indices))
        for batch_start in range(0, len(data), train_batch_size):
            batch_end = min(batch_start+train_batch_size, len(data))
            X_batch = data[batch_start: batch_end]
            y_batch = labels[batch_start: batch_end]
            sents_tensor, masks_tensor, sents_lengths = self.convert_sents_to_ids_tensor(X_batch)
            output = self.model(input_ids = sents_tensor.to(self.device), attention_mask = masks_tensor.to(self.device))[0]
            self.model.zero_grad()
            
            loss = loss_func(output, y_batch)
            train_loss += loss.item()
            loss.backward()

            # clip the norm of the gradients to 1, to prevent exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # update learning rate
            self.scheduler.step()

        return train_loss/n_batches
            

    def cross_validate(self, n_folds = 5, output="", test_size = 0.3, train_batch_size = 64, n_epochs=20, test_batch_size = 128, class_weights = None, out = "", patience=3, idx_label_map={}, best_epoch_temp=""):
        kfold = StratifiedShuffleSplit(n_splits = n_folds, test_size = test_size, random_state=self.random_state)
        fold = 0
        with open(output, "w") as f:
            f.write(".........................................\n")
        for train_indices, test_indices in kfold.split(self.data, self.labels):
            X_train = self.data[train_indices]
            y_train = self.labels[train_indices]
            X_remaining = self.data[test_indices]
            y_remaining = self.labels[test_indices]
            print("Fold: {}........................................".format(fold))
      
            fold+=1

            # X_valid = X_remaining.copy()
            # X_test = X_remaining.copy()
            # y_valid =  y_remaining.copy()
            # y_test = y_remaining.copy()

            X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size =0.5, random_state=self.random_state, stratify=y_remaining)

            self.model = BertForSequenceClassification.from_pretrained(self.bert_config, num_labels = self.num_classes)
            self.model.to(self.device)
            n_batches = int(np.ceil(len(X_train)/train_batch_size))
            print("Number of batches: ", n_batches)
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
            total_step = n_batches * 20
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                    num_warmup_steps=0,
                                                                    num_training_steps=total_step)
            if class_weights!=None: 
                class_weights = torch.tensor([class_weights[k] for k in sorted(class_weights.keys())])
                
                criterion = torch.nn.CrossEntropyLoss(weight = class_weights.to(self.device), reduction = 'mean')
            else:
                criterion = torch.nn.CrossEntropyLoss()
            self.model.train()
            y_train = torch.tensor(y_train, dtype = torch.long).to(self.device)
            valid_loss_hist = []
            epoch = 0
            best_epoch = 0
            while True:
                begin_time = time.time()
                train_loss = self.fit(X_train, y_train, train_batch_size, criterion)

                #validation: 
                valid_loss, valid_outputs = self.predict(X_valid, y_valid, criterion, test_batch_size, val=True)
                print("Epoch: {}, train loss: {}. valid loss: {}, time: {}".format(epoch, train_loss, valid_loss, time.time()-begin_time))
                
                improved_loss = len(valid_loss_hist)==0 or valid_loss < min(valid_loss_hist)
                valid_loss_hist.append(valid_loss)
                epoch+=1
                if improved_loss:
                    best_epoch = epoch
                    # self.model.save_pretrained(best_epoch_temp)
                    # # save best model
                    # print('Save bew best model....., epoch: ', epoch)
                
                else: #if valid loss did not improve
                    if (epoch > n_epochs):
                        break
                
            print("Training process ends!!!!")

            
            # self.model = BertForSequenceClassification.from_pretrained(best_epoch_temp)
            # self.model.to(self.device)
            test_loss, y_pred = self.predict(X_test, y_test, criterion, batch_size = test_batch_size)
            print(confusion_matrix(y_test, y_pred, normalize = 'true'))
            print("Acc: ", accuracy_score(y_test, y_pred))
            print("Classification report: ", classification_report(y_test, y_pred, target_names =  [idx_label_map[key] for key in range(len(idx_label_map))]))
            with open(output, "a") as f:
                f.write("Fold: {}\n".format(fold))
                f.write(classification_report(y_test, y_pred, target_names =  [idx_label_map[key] for key in range(len(idx_label_map))]))
                f.write("\n.........................................\n")

            # shutil.rmtree(best_epoch_temp)

    def train(self, test_size = 0.15, train_batch_size = 64, test_batch_size = 128, n_epochs = 5, class_weights = None):
         # train, test split
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size = test_size, random_state = self.random_state, stratify = self.labels)
        self.model = BertForSequenceClassification.from_pretrained(self.bert_config, num_labels = self.num_classes)
        self.model.to(self.device)
        n_batches = int(np.ceil(len(X_train)/train_batch_size))
        print("Number of batches: ", n_batches)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_step = n_batches * n_epochs
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=total_step)
        
        if class_weights!=None: 
            class_weights = torch.tensor([class_weights[k] for k in sorted(class_weights.keys())])
            
            criterion = torch.nn.CrossEntropyLoss(weight = class_weights.to(self.device), reduction = 'mean')
        else:
            criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        y_train = torch.tensor(y_train, dtype = torch.long).to(self.device)

        for epoch in range(n_epochs):
            begin_time = time.time()
            loss = self.fit(X_train, y_train, train_batch_size, criterion)
            print("Epoch: {}, training loss: {}, time: {}".format(epoch, loss, time.time()-begin_time))

        print("Training process ends!!!!")

        test_loss, outputs = self.predict(X_test, y_test, criterion, batch_size = test_batch_size)
        return y_test, outputs

  
   
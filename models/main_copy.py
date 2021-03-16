import os 
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
print(path)
sys.path.append(path)
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import csv
import re
import pandas as pd 
from utils.help_functions import text_to_exp_labels
from utils.tweet_preprocessing import tokenizeRawTweetText
from tfidf_cls import TfidfModel
from bert_cls import Bertweet
from bert_mtl import BertMLTModel
from config.config import Config
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from scipy.sparse import vstack
import torch
import random
# import these modules 
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
random.seed(12345)
np.random.seed(67891)
torch.manual_seed(54321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def read_data(input_path, selected_columns = None, delimiter = ",", text_col = "", is_return_files=False):
    # read data
    
    if is_return_files:
        data = []
    else:
        data = pd.DataFrame()
    count = 0
    if os.path.isfile(input_path):
        data = pd.read_csv(input_path, delimiter = delimiter)
    else:
        for dir, _, filenames in os.walk(input_path):
            while len(filenames) > 0:
                file = filenames.pop()
                if file.endswith(".csv"):
                    print("file: ", file)
                    temp_df = pd.read_csv(os.path.join(dir, file), delimiter=delimiter)
                    if selected_columns!=None: #otherwise select all columns
                        
                        temp_df = temp_df[selected_columns]
                        if text_col!= "":
                            # temp_df[Config.text_col] = temp_df[Config.text_col].apply(lambda x: re.sub("^\'|\'$|^\"|\"$", '', x))
                            temp_df[Config.prepro_text] = temp_df[Config.text_col].apply(lambda x: tokenizeRawTweetText(x))
                            temp_df[Config.prepro_text] = temp_df[Config.prepro_text].apply(lambda x: re.sub(" +", ' ', x))
                            print("Temp: ", temp_df.columns)
                    if is_return_files:
                        data.append(temp_df)
                    else:
                        data = pd.concat([data, temp_df])

    return data

if __name__ == "__main__":

    
    data = read_data(Config.input_path, Config.selected_columns, text_col = Config.text)
    print(data.columns)
    #merging some categories
    if Config.label_type_map!=None:
        data[Config.label] = data[Config.label].apply(lambda x: Config.label_type_map[x])

    data[Config.explan] = data[Config.explan].fillna(value ='')
    data[Config.prepro_explan] = data[Config.explan].apply(lambda x: x.strip().replace('\n', Config.sep_explan_token))
    data[Config.prepro_text] = data[Config.text].apply(lambda x: tokenizeRawTweetText(x))
    data[Config.prepro_text] = data[Config.prepro_text].apply(lambda x: re.sub(" +", ' ', x))
    data[Config.prepro_explan] = data[Config.prepro_explan].apply(lambda x: tokenizeRawTweetText(x))
    data[Config.prepro_explan] = data[Config.prepro_explan].apply(lambda x: re.sub(" +", " ", x))
    data[Config.prepro_explan] = data[Config.prepro_explan].apply(lambda x: [y.strip() for y in x.split(Config.sep_explan_token)])
    data['len'] = data[Config.prepro_text].apply(lambda x: len(x.split(" ")))
    data = data[data['len'] >= 3]
    print("Data size after removing short tweets: ", data.shape)
    labels = data[Config.label].value_counts()

    label_idx_map = {key: i for i, key in enumerate(labels.keys())}
    idx_label_map = {value: key for key, value in label_idx_map.items()}
    data[Config.prepro_label] = data[Config.label].apply(lambda x: label_idx_map[x])
    
    max_classes =  float(max(dict(labels).values()))
    label_count = dict(data[Config.prepro_label].value_counts())
    if Config.add_extra_data:
        data_extra = read_data(Config.extra_labeled_path, selected_columns=[Config.id, Config.text, Config.label])
        
        if Config.extra_drop_row !="":
            data_extra = data_extra[data_extra[Config.label]!=Config.extra_drop_row]
        
        
        data_extra.reset_index(drop = True, inplace=True)
        print(data_extra.shape, len(set(data_extra[Config.id]))) 
        #merging some categories
        if Config.label_type_map!=None:
            data_extra[Config.label] = data_extra[Config.label].apply(lambda x: Config.label_type_map[x])
        print(data_extra.head(n=5))
        data_extra[Config.prepro_text] = data_extra[Config.text].apply(lambda x: tokenizeRawTweetText(x))
        data_extra[Config.prepro_text] = data_extra[Config.prepro_text].apply(lambda x: re.sub(" +", ' ', x))
        data_extra[Config.prepro_label] = data_extra[Config.label].apply(lambda x: label_idx_map[x])
        data_extra[Config.label].describe()
        print("Extra: ", data_extra.shape)
        print("Initial Data: ", data.shape)
        data = pd.concat([data, data_extra])


    dataxx = pd.DataFrame()
    for label in set(data[Config.label]):
        if str(label) == "not_related_or_irrelevant":
            dataxx = pd.concat([dataxx, data[data[Config.label] == label][0:500]])
        else:
            dataxx = pd.concat([dataxx, data[data[Config.label]==label]])
    data = dataxx.copy()
    data.reset_index(drop=True, inplace=True)
    print("Final data: ", data.shape)
    data.to_csv(Config.data_final_path, index=False)
    data = pd.read_csv(Config.data_final_path)
    print(data[Config.label].value_counts())
    print(len(set(data[Config.id])))


    # sys.exit()
    label_count = dict(data[Config.prepro_label].value_counts())
    if Config.cls_weights == 'log':
        class_weights = {i:np.log(label_count[i]) for i in range(len(label_count))}
    elif Config.cls_weights == 'balanced':
        class_weights = {i: data.shape[0]/(len(label_count)*label_count[i]) for i in range(len(label_count))}
    elif Config.cls_weights =='max_class':
        class_weights = {i: max_classes/label_count[i] for i in range(len(label_count))}
    else:
        class_weights = None
    print("Class weights: ", class_weights)
    svc = SVC(random_state = Config.random_state, class_weight = class_weights)
    model = None
    if Config.tfidf_cls:
        tfidf_text = Config.prepro_text
        if Config.tfidf_lemmatizer:
            lemmatizer  = WordNetLemmatizer()
            tfidf_text = 'prepro_tfidf'
            stop_words = set(stopwords.words('english'))
            data[tfidf_text] = data[Config.prepro_text].apply(lambda x: ' '.join([lemmatizer.lemmatize(y) for y in x.split(" ")]))
            data[tfidf_text] = data[tfidf_text].apply(lambda x: ' '.join([y for y in x.split(" ") if y not in stop_words]))
            

        model = TfidfModel(svc, list(data[tfidf_text]), list(data[Config.prepro_label]), ngram_range = Config.tfidf_ngram_range, min_df = Config.tfidf_min_df)
        
        # print("Cross validate......")
        # y_true, y_pred = model.cross_validate(cv = 5, grid_params = grid_params)
        # print("Confusion matrix: ")
        # print(confusion_matrix(y_true, y_pred, normalize = 'true'))
        # print("Acc: ", accuracy_score(y_true, y_pred))
        

        print(".....................")
        print("Train/test split")
        y, y_pred = model.train(test_size = Config.test_size, param_grid = Config.tfidf_grid_params)
       

        print(confusion_matrix(y, y_pred, normalize = 'true'))
        print("Acc: ", accuracy_score(y, y_pred))
        print("Classification report: ", classification_report(y, y_pred,  target_names = [idx_label_map[key] for key in range(len(idx_label_map))]))
        print("....Train:")
        y_pred_train = model.predict(list(data[tfidf_text]),require_feature_extraction=True)
        print("F1: ", f1_score(list(data[Config.prepro_label]), y_pred_train, average = 'macro'))
        print("ACC:", accuracy_score(list(data[Config.prepro_label]), y_pred_train))
    if Config.bertweet: 
        model = Bertweet(list(data[Config.prepro_text]), list(data[Config.prepro_label]), bert_config =Config.bert_config,device = Config.device)
        model.cross_validate(idx_label_map=idx_label_map, n_folds = Config.n_folds)
        # y, y_pred = model.train(test_size = Config.test_size, n_epochs = Config.bert_seq_epochs, class_weights = class_weights)
        # print(confusion_matrix(y, y_pred, normalize = 'true'))
        # print("Acc: ", accuracy_score(y, y_pred))
        # print("Classification report: ", classification_report(y, y_pred, target_names =  [idx_label_map[key] for key in range(len(idx_label_map))]))
    
    if Config.bert_mtl:
        model = BertMLTModel(list(data[Config.prepro_text]), list(data[Config.prepro_label]), list(data[Config.prepro_explan]),
                        bert_config = Config.bert_config, device=Config.device, cls_hidden_size = Config.cls_hidden_size, 
                        exp_hidden_size = Config.exp_hidden_size, random_state = Config.random_state)
    
        model.train(test_size = Config.test_size, n_epochs = Config.bert_mtl_epochs, cls_class_weights = class_weights)
        
    
    if Config.tfidf_cls and Config.gather_data:
        gether_data(model, text_col = Config.prediction_data_text)
        

               

                
                    
    

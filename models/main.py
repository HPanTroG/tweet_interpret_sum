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
from utils.help_functions import gather_data
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
                    if is_return_files:
                        data.append(temp_df)
                    else:
                        data = pd.concat([data, temp_df])

    return data

def extract_files(folder):
    files = []
    if os.path.isfile(folder):
        files = [folder]
    else:
        for dir, _, filenames in os.walk(folder):
            while len(filenames) > 0:
                file = filenames.pop()
                if file.endswith(".csv"):
                    files.append(os.path.join(dir, file))

    return files

if __name__ == "__main__":

    
    data = read_data(Config.input_path, Config.selected_columns, text_col = Config.text)
    
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
    print(data.shape)
    # data = data[data['len'] >= 3]
    # print("Data size after removing short tweets: ", data.shape)
    labels = data[Config.label].value_counts()

    label_idx_map = {key: i for i, key in enumerate(labels.keys())}
    idx_label_map = {value: key for key, value in label_idx_map.items()}
    data[Config.prepro_label] = data[Config.label].apply(lambda x: label_idx_map[x])
    
    max_classes =  float(max(dict(labels).values()))
    label_count = dict(data[Config.prepro_label].value_counts())
    print("............")
    print(data[Config.label].value_counts())
   
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
        model.cross_validate(idx_label_map=idx_label_map, n_folds = Config.n_folds, output=Config.bert_seq_output, best_epoch_temp = Config.best_epoch_temp, patience=Config.patience)
        # y, y_pred = model.train(test_size = Config.test_size, n_epochs = Config.bert_seq_epochs, class_weights = class_weights)
        # print(confusion_matrix(y, y_pred, normalize = 'true'))
        # print("Acc: ", accuracy_score(y, y_pred))
        # print("Classification report: ", classification_report(y, y_pred, target_names =  [idx_label_map[key] for key in range(len(idx_label_map))]))
    
    if Config.bert_mtl:
        model = BertMLTModel(list(data[Config.prepro_text]), list(data[Config.prepro_label]), list(data[Config.prepro_explan]),
                        bert_config = Config.bert_config, device=Config.device, cls_hidden_size = Config.cls_hidden_size, 
                        exp_hidden_size = Config.exp_hidden_size, random_state = Config.random_state)

        model.cross_validate(idx_label_map=idx_label_map, n_epochs = Config.bert_mtl_epochs, best_model_path = Config.best_mtl_path, n_folds = Config.n_folds, train_batch_size = Config.train_batch_size,
                        test_batch_size = Config.test_batch_size, out=Config.bert_mlt_output, test_size = Config.test_size * 2, cls_weights=class_weights, exp_weights = Config.exp_weights,
                        patience = 3, best_epoch_temp = Config.best_epoch_temp)
    
        # model.train(test_size = Config.test_size, n_epochs = Config.bert_mtl_epochs, cls_weights = class_weights, exp_weight = Config.exp_weights[0], idx_label_map= idx_label_map, model_path = Config.model_path)
        
        # model.check_explanation(saved_model = Config.best_mtl_path,idx_label_map=idx_label_map, batch_size = Config.test_batch_size)
    if Config.new_data_prediction:
        print("Input folder: ", Config.new_data_path)
        files = extract_files(Config.new_data_path)

        for file in files:
            print("file: {}........".format(file))
            new_data = pd.read_csv(file, delimiter="\t")
            output_path = Config.classified_new_data_path+file[file.rindex("/"):]
            print("data size: {}".format(new_data.shape))

            new_data = new_data[['tweet_id', 'created_at', 'text']]
            new_data.columns  = [Config.id, 'created_at', Config.text]
            new_data[Config.prepro_text] = new_data[Config.text].apply(lambda x: tokenizeRawTweetText(x))
            new_data[Config.prepro_text] = new_data[Config.prepro_text].apply(lambda x: re.sub(" +", ' ', x))
            new_data['len'] = new_data['prepro_text'].apply(lambda x: len(x.split(" ")))
            new_data = new_data[new_data['len']>1]
            print("data size (after removing very short tweet len<=1): {}".format(new_data.shape))
            model = BertMLTModel(list(new_data[Config.prepro_text]), None, None,
                        bert_config = Config.bert_config, device=Config.device, cls_hidden_size = Config.cls_hidden_size, 
                        exp_hidden_size = Config.exp_hidden_size, random_state = Config.random_state, n_cls_classes = len(label_idx_map))
            model.classify_new_data(model_path= Config.model_path, input = {'id': np.array(new_data[Config.id]), 'created_at': np.array(new_data['created_at']),
                            'text': np.array(new_data[Config.text]), 'prepro_text': np.array(new_data[Config.prepro_text])}, 
                            output_path = output_path, batch_size = Config.test_batch_size, idx_label_map= idx_label_map)

            


    if Config.tfidf_cls and Config.gather_data:
        gether_data(model, text_col = Config.prediction_data_text)
        

               

                
                    
    

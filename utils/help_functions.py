import re
import os 
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)    
from config.config import Config   
from utils.tweet_preprocessing import tokenizeRawTweetText

def text_to_exp_labels(org_text, explan_text):
    """
        param org_text: str, original text
        param explan_text: str, explan_text
        @return list[int], list of 0/1 values to label whether each token in org_text is a part of explan_text 
    """
    labels = org_text
    for chunk in explan_text.split("\n"):
        labels = re.sub(chunk, '1 '*len(chunk.split( )), labels)

    labels = re.sub('[^1 ]', '0', labels)
    labels = re.sub('  ', ' ', labels).strip().split(" ")
    labels = [1 if i == '1' else 0 for i in labels]
    return labels

# text_to_exp_labels("The programming is a new programming', 'programming\nprogramming')

def gather_data(model, data, text_col = ""):
    prediction_datas = read_data(Config.prediction_file, delimiter=Config.prediction_file_delimiter, text_col = text_col, is_return_files=True)
    is_loop = True
    file_id = 0
    new_labeled_data = {}
    
    # uncomment to continue collecting data from the previous collected data
    # data2 = pd.read_csv(Config.extra_unverified_label_path, delimiter="\t", quoting=csv.QUOTE_NONE)
    # data2[Config.text] = data2[Config.text].apply(lambda x: re.sub("^\'|\'$", '', x))
    # data2[Config.prepro_text] = data2[Config.text].apply(lambda x: tokenizeRawTweetText(x))
    # data2[Config.prepro_text] = data2[Config.prepro_text].apply(lambda x: re.sub(" +", ' ', x))
    # data2.rename(columns={Config.label:Config.prepro_label}, inplace=True)
    # print("Before: ", data.shape, data2.shape)
    # data = pd.concat([data, data2])
    # print(data[[Config.label, Config.prepro_label, Config.id]].head(n=4))
    
    # print("Concatenated shape:", data.shape)
    # print("Class needs to more data: ")
    # label_count = dict(data2[Config.prepro_label].value_counts())

    label_count = dict(data[Config.prepro_label].value_counts())
    for label, count in label_count.items():
        # if count< Config.num_extra_data:
        if count<Config.num_data_required:
            new_labeled_data[label] = model.feature_extraction(data[data[Config.prepro_label]==label][Config.prepro_text])
            print(label, idx_label_map[label], count)
    
    # collected = dict(data2[Config.prepro_label].value_counts())
    # loop = {key: collected[key] for key in new_labeled_data.keys()}
    loop = {key: 0 for key in new_labeled_data.keys()}
    print("Collected: ", loop)
    # input()
    with open(Config.extra_unverified_label_path, "w") as f: 
        f.write("{}\t{}\t{}\n".format(Config.id, Config.text, Config.label))
    sim = 0
    notin=0
    while is_loop == True and file_id<len(prediction_datas):
        prediction_data = prediction_datas[file_id]
        file_id+=1
        print("data: {}, file_id: {}".format(prediction_data.shape, file_id))
        
     
        batch_size = 100
        
        for batch_start in range(0, prediction_data.shape[0], batch_size):
            batch_end = min(batch_start+batch_size, len(prediction_data))
            
            batch_features = model.feature_extraction(np.array(prediction_data.iloc[batch_start:batch_end][Config.prepro_text]))

            predicted_labels = model.predict(batch_features)
            for i, label in enumerate(predicted_labels):
                line = i+batch_start
                if label not in new_labeled_data:
                    notin+=1
                    print(label, "notin: ", notin)
                    continue
                elif loop[label]> Config.num_extra_data:
                    loop[label]+=1
                    count=0
                    for key, value in loop.items():
                        if value>= Config.num_extra_data:
                            count+=1
                    if count==len(loop): 
                        is_loop = False
                        break
                else: # ignore tweets that are very similar to chosen ones
                    if sum(cosine_similarity(batch_features[i], new_labeled_data[label])[0]>=Config.tweet_cosine_sim_thres) >=1:
                        sim+=1
                        print(label, "sim:", sim)
                        continue
                    with open(Config.extra_unverified_label_path, "a") as f:
                        f.write("{}\t{}\t{}\n".format(prediction_data.iloc[line][Config.id], 
                        prediction_data.iloc[line][Config.text], label))
                        new_labeled_data[label] = vstack([new_labeled_data[label], batch_features[i]])
                        loop[label]+=1
    for key, value in loop.items():
        print("label:{}--{}, collected: {}".format(key, idx_label_map[key], value))

def add_extra_data(data):
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

    return data
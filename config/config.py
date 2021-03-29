class Config:
    working_directory = "/home/nguyen/tweet_interpret_sum/datasets/"
    # data_name = "2015_Nepal_Earthquake_en_CF_labeled_data_final.csv"
    data_folder = "2014_Philippines_Typhoon_Hagupit_en"
    data_name = "2014_Philippines_Typhoon_Hagupit_en_CF_labeled_data_final.csv"
    input_path = working_directory+"labeled_data/"+data_name
    
    # if extra data needed
    prediction_file = working_directory +"unlabeled_data/2015_Nepal_Earthquake_en/crawled_data/"
    extra_unverified_label_path = working_directory + "extra_labeled_data/unverified/typhoon_hagupit1.csv"
    extra_labeled_path = working_directory+"labeled_data/2015_Nepal_Earthquake_en_CF_labeled_data_extra.csv"
    data_final_path = working_directory+"/labeled_data/2015_Nepal_Earthquake_en_CF_labeled_data_final.csv"
    extra_drop_row = 'not_verified'
    prediction_data_columns = []
    prediction_file_delimiter = "\t"
    num_extra_data = 1000
    num_data_required= 400    
    tweet_cosine_sim_thres = 0.8
    prediction_data_text = 'text'
    
    
    # merge labels in input data due to the shortage of instances in some classes
    label_type_map = {'not_related_or_irrelevant':'not_related_or_irrelevant',
                    'rescue_volunteering_and_donation_effort':'rescue_volunteering_and_donation_effort',
                    'injured_or_dead_people':'injured_or_dead_people',
                    'displaced_people_and_evacuations':'affected_people_and_evacuations',
                    'infrastructure_and_utilities_damage':'infrastructure_and_utilities_damage',
                    'caution_and_advice':'caution_and_advice',
                    'missing_trapped_or_found_people':'affected_people_and_evacuations',
                    'other_useful_information':'other_useful_information',
                    'affected_people_and_evacuations': 'affected_people_and_evacuations'}
    
 
    # general settings
    id = 'tweet_id'
    text = 'tweet_text'
    label = 'corrected_label'
    explan = 'informative_content'
    prepro_text = 'prepro_text'                    
    prepro_explan= 'prepro_explan'
    prepro_label = 'prepro_label'
    selected_columns = [id, text, label, explan]
    sep_explan_token  = " _sep_exp_token_ "
    bert_config = 'vinai/bertweet-base'
    random_state = 12
    test_size = 0.15
    device = 'cuda'
    cls_weights = None
    gather_data = False
    add_extra_data = False
    train_batch_size = 64
    test_batch_size = 128
    n_folds = 5
    best_epoch_temp= working_directory +"temps/" + data_name[:-4]+".temp"
    patience = 3

    # for tfidf model
    tfidf_grid_params = {'C':[i/10 for i in range(1, 10)]}
    tfidf_lemmatizer = True
    tfidf_min_df = 1
    tfidf_ngram_range = (1, 1)
    


    # bert for sequence classification
    bert_seq_epochs = 20
    bert_seq_output = working_directory +"results/"+data_name[:-4]+"_bertseq1.csv"
    

    #for bert mtl
    cls_hidden_size = 768
    exp_hidden_size = 768
    bert_mtl_epochs = 20
    # exp_weights = [i/1000 for i in range(1, 10)]+[i/100 for i in range(1, 10)]
    exp_weights=[0]
    best_mtl_path = working_directory+"saved_models/"+data_name[:-4]+"_noexp.json"
    bert_mlt_output = working_directory+"results/"+ data_name[0:-4]+"_noexp.csv"

    # new data prediction path
    model_path = working_directory+"saved_models/"+data_name[0:-4]+"_best.json"
    new_data_path = working_directory + "unlabeled_data/"+data_folder+"/crawled_data/150425104337_nepal_earthquake_20150427_vol-17.json.csv" 
    classified_new_data_path = working_directory + "unlabeled_data/"+ data_folder+"/mtl_classified_data/"
   

    # choose model to run
    tfidf_cls =False
    bertweet = False
    bert_mtl = True
    new_data_prediction = False


    
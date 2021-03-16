class Config:
    working_directory = "/home/nguyen/tweet_interpret_sum/datasets/"
    input_path = working_directory+"labeled_data/2015_Nepal_Earthquake_en_CF_labeled_data_final.csv"

    prediction_file = working_directory +"unlabeled_data/2015_Nepal_Earthquake_en/crawled_data/"
    extra_unverified_label_path = working_directory + "extra_labeled_data/unverified/typhoon_hagupit1.csv"
    extra_labeled_path = working_directory+"labeled_data/2015_Nepal_Earthquake_en_CF_labeled_data_extra.csv"
    data_final_path = working_directory+"/labeled_data/2015_Nepal_Earthquake_en_CF_labeled_data_final.csv"
    extra_drop_row = 'not_verified'
    prediction_data_columns = []
    prediction_file_delimiter = "\t"
    num_extra_data = 1000
    num_data_required= 400    
    tweet_cosine_sim_thres = 0.85
    prediction_data_text = 'text'
    n_folds = 5
    
    
    label_type_map = {'not_related_or_irrelevant':'not_related_or_irrelevant',
                    'rescue_volunteering_and_donation_effort':'rescue_volunteering_and_donation_effort',
                    'injured_or_dead_people':'injured_or_dead_people',
                    'displaced_people_and_evacuations':'affected_people_and_evacuations',
                    'infrastructure_and_utilities_damage':'infrastructure_and_utilities_damage',
                    'caution_and_advice':'caution_and_advice',
                    'missing_trapped_or_found_people':'affected_people_and_evacuations',
                    'other_useful_information':'other_useful_information',
                    'affected_people_and_evacuations': 'affected_people_and_evacuations'}
    final_class = ['not_related_or_irrelevant', 'rescue_volunteering_and_donation_effort',
                'injured_or_dead_people', 'affected_people_and_evacuations', 'other_useful_information',
                'infrastructure_and_utilities_damage']
    # label_type_map = None
 

    id = 'tweet_id'
    text = 'informative_content'
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
    device = 'cuda:2'
    cls_weights = None
    gather_data = False
    add_extra_data = False

    # for tfidf model
    tfidf_grid_params = {'C':[i/10 for i in range(1, 10)]}
    tfidf_lemmatizer = True
    tfidf_min_df = 1
    tfidf_ngram_range = (1, 1)
    


    # bert for sequence classification
    bert_seq_epochs = 20

    #for bert mtl
    cls_hidden_size = 768
    exp_hidden_size = 768
    bert_mtl_epochs = 20


    tfidf_cls =False
    bertweet = True
    bert_mtl = False
    
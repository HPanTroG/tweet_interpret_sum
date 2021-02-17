class Config:
    working_directory = "/home/nguyen/tweet_interpret_sum/datasets/"
    input_path = working_directory+"labeled_data/2014_Pakistan_floods_CF_labeled_data.csv"

    id = 'tweet_id'
    text = 'tweet_text'
    label = 'corrected_label'
    explan = 'informative_content'
    prepro_text = 'prepro_text'
    prepro_explan= 'prepro_explan'
    prepro_label = 'prepro_label'
    selected_columns = [id, text, label, explan]

    random_seed = 12
    test_size = 0.1
    device = 'cuda:2'
    
import sys
import os
import pandas as pd 


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
    folder = '/home/nguyen/tweet_interpret_sum/datasets/unlabeled_data/2015_Nepal_Earthquake_en/mtl_classified_data/'
    files = extract_files(folder)
    with open(folder+'statistics.txt', "w") as f:
        f.write("")
    for file in files:
        data = pd.read_csv(file, delimiter = '\t')
        with open(folder +"statistics.txt", "a") as f:
            f.write(file[file.rindex("/")+1:]+"\n")
            for key, value in dict(data['predicted_label'].value_counts()).items():
                f.write("%-50s%10d%10.2f\n"%(key, value, value/data.shape[0]))
            f.write(".......................................................\n")

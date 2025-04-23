


import os
import sys
import math
import pandas as pd

ext = '.csv'
directory = "E:/UCF_studies/Spring 2025/CAP 5610 - Machine Learning/project/explainable-AI-for-genomic-tasks/SHAP_NMT/SHAP_exp/results/"
dataset_path = "E:/UCF_studies/Spring 2025/CAP 5610 - Machine Learning/project/explainable-AI-for-genomic-tasks/SHAP_NMT/dataset.csv"

if len(sys.argv)>1:
    directory = sys.argv[1]
if len(sys.argv)>2:
    directory = sys.argv[2]



tatabox = "TATA"

def read_files_in_subdirectory(directory, file_extension='.csv'):
    

    dataset = pd.read_csv(dataset_path)

    total_tokens = 0
    tata_tokens = 0
    files_read = 0
    positive_token = 0
    negative_token = 0
    positive_tata = 0
    negative_tata = 0
    tata_acc = 0
    tata_non_acc = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                # print(f"\nReading file: {file_path}")
                label = int(dataset.iloc[int(file.split('.')[0])]["label"])
                # print(label)
                files_read += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        next(f)  # Skip the header
                        for line in f:
                            token = line.strip().split(',')[0]
                            value = float(line.strip().split(',')[1])
                            positive = False
                            if token and value:
                                # print(token, end = '\t')
                                if value >= 0:
                                    positive=True
                                    positive_token += 1
                                if tatabox in token:
                                    # print("present")
                                    tata_tokens += 1
                                    if positive:
                                        positive_tata += 1
                                        tata_acc += label
                                        tata_non_acc += 1-label
                                total_tokens += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    # print(line.split(',')[1])
                    # break

    negative_token = total_tokens - positive_token
    negative_tata = tata_tokens - positive_tata

    return tata_tokens, total_tokens, files_read, positive_token, negative_token, positive_tata, negative_tata, tata_acc, tata_non_acc




def calculate_stats():

    tata_tokens, total_tokens, files_read, positive_token, negative_token, positive_tata, negative_tata, tata_acc, tata_non_acc = read_files_in_subdirectory(directory, file_extension=ext)

    print("tata_tokens = ", tata_tokens, "\ntotal_tokens = ", total_tokens)
    print("Total files read = ", files_read)
    print("tata percentage = ", 100*tata_tokens/total_tokens, "%")

    total_seq = 5930
    scale = total_seq/files_read
    mod_tata_tokens = math.floor(tata_tokens*scale)
    mod_total_tokens = math.floor(total_tokens*scale)
    mod_positive_tokens = math.floor(positive_token*scale)
    mod_negative_tokens = math.floor(negative_token*scale)
    mod_positive_tata = math.floor(positive_tata*scale)
    mod_negative_tata = math.floor(negative_tata*scale)
    mod_tata_acc = math.floor(tata_acc*scale)
    mod_tata_non_acc = math.floor(tata_non_acc*scale)

    print("*"*100)
    print("tata_tokens = ", mod_tata_tokens, "\ntotal_tokens = ", mod_total_tokens)
    print("Total files read = ", total_seq, "scale=", scale)

    print("positive tokens = ", mod_positive_tokens)
    print("negative tokens = ", mod_negative_tokens)
    print("positive tata = ", mod_positive_tata)
    print("negative tata = ", mod_negative_tata)
    
    print("accurate tata = ", mod_tata_acc)
    print("non accurate tata = ", mod_tata_non_acc)

    return scale



token_dict = {}

def set_token_value(token, value=1):
    if token not in token_dict.keys():
        token_dict[token]=0

    token_dict[token] = token_dict[token]+value



def max_tokens(directory, file_extension='.csv'):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        next(f)  # Skip the header
                        for line in f:
                            token = line.strip().split(',')[0]
                            value = float(line.strip().split(',')[1])
                            if token=="TCTATA":
                                print(file_path)
                                # exit(0)
                            if token and value:
                                if value >= 0:      # only consider positive importance
                                    set_token_value(token, value)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")





def print_top_n(input_dict, n=10, scale=1):
     # Sort the dictionary by value in descending order
    sorted_items = sorted(input_dict.items(), key=lambda item: item[1], reverse=True)

    # Print the top n items
    for key, value in sorted_items[:n]:
        print(f"{key}: {math.floor(value*scale)}")



def print_top_n_tokens(n_tokens=10, scale=1):
    max_tokens(directory, file_extension=ext)

    print(f"Top {n_tokens} tokens with positive importance")
    print_top_n(token_dict, n=n_tokens, scale=scale)


if __name__ == "__main__":
    scale = 1
    scale = calculate_stats()

    print("*" * 100)

    # print_top_n_tokens(n_tokens=25, scale=scale)

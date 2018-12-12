import sys
import os
import math
import re
import string
from random import randint
import json
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import normalize
from scipy import sparse

tw_tknzr=TweetTokenizer(strip_handles=True, reduce_len=True)
cache_english_stopwords=stopwords.words('english')
stemmer = PorterStemmer()

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tweet_clean(tweet):
    # Remove tickers
    sent_no_tickers=re.sub(r'\$\w*','',tweet)
    
    temp_tw_list = tw_tknzr.tokenize(sent_no_tickers)

    # Remove stopwords
    list_no_stopwords=[i for i in temp_tw_list if i.lower() not in cache_english_stopwords]

    # Remove hyperlinks
    list_no_hyperlinks=[re.sub("\d+", "", i) for i in list_no_stopwords]

    # Remove Punctuation and split 's, 't, 've with a space for filter
    list_no_punctuation=[re.sub(r'['+string.punctuation+']+', ' ', i) for i in list_no_hyperlinks]

    # Remove multiple whitespace
    new_sent = ' '.join(list_no_punctuation)
    # Remove any words with 2 or fewer letters
    filtered_list = tw_tknzr.tokenize(new_sent)
    list_filtered = [re.sub(r'^\w\w?$', '', i) for i in filtered_list]

    return stem_tokens(list_filtered)


def main(filtered_city_dir, cityname, bow_dir):
    if not os.path.exists(bow_dir):
        os.makedirs(bow_dir)

    if not os.path.exists(bow_dir+"/"+cityname):
        os.makedirs(bow_dir+"/"+cityname)
    vocabulary = []
    business_list = {}
    reviews = []
    result = []
    business_split = {}
    i=0
    input_file = open(filtered_city_dir+"/review_"+cityname+".json", "r")
    data = json.loads(input_file.read())
    for line in data:
        text = line["text"]
        if line["business_id"] not in business_list:
            business_list[line["business_id"]] = i
            i+=1
        business_id = business_list[line["business_id"]]
        if business_id not in business_split:
            business_split[business_id] = {"labels": [], "text": []}
        business_split[business_id]["labels"].append([business_id, line["useful"], line["review_count"]])
        
        tokens = tweet_clean(text)
        temp = ' '.join(tokens)
        clean_sent=re.sub(r'\s\s+', ' ', temp)
        clean_sent=clean_sent.lstrip(' ')
        business_split[business_id]["text"].append(clean_sent)
        vocabulary.extend(tokens)
    input_file.close()

    vocabulary = set(vocabulary)
    print(len(vocabulary))
    print("finish collecting vocabulary")

    vectorizer = CountVectorizer(vocabulary=vocabulary)

    test_set = None
    train_set = None
    test_idx = 0
    train_idx = 0
    for business, features in business_split.items():
        if business % 500 == 0:
            print("spliting on: "+str(business))
        labels = np.array(features["labels"])
        text = features["text"]
        random = randint(0,4)
        if random == 0 and test_set is not None and len(test_set)+len(text) > 1000:
            print("save test")
            test_csr_matrix = sparse.csr_matrix(test_set)
            sparse.save_npz(bow_dir+"/"+cityname+"/"+"test_"+str(test_idx)+".npz", test_csr_matrix)
            test_set = None
            test_idx+=1
        elif random != 0 and train_set is not None and len(train_set)+len(text) > 1000:
            print("save train")
            train_csr_matrix = sparse.csr_matrix(train_set)
            sparse.save_npz(bow_dir+"/"+cityname+"/"+"train_"+str(train_idx)+".npz", train_csr_matrix)
            train_set = None
            train_idx+=1
        
        tokens = vectorizer.transform(text).toarray()
        normed_tokens = normalize(tokens, axis=1, norm='l1')
        print(normed_tokens.shape)
        output = np.hstack((labels, normed_tokens))
        if random == 0:
            if test_set is None:
                test_set = output
            else: 
                test_set = np.concatenate((test_set,output))
        else:
            if train_set is None:
                train_set = output
            else: 
                train_set = np.concatenate((train_set,output))

    sparse.save_npz(bow_dir+"/"+cityname+"/"+"train.npz", sparse.csr_matrix(train_set))
    sparse.save_npz(bow_dir+"/"+cityname+"/"+"test.npz", sparse.csr_matrix(test_set))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
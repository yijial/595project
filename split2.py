import sys
import os
import math
import re
import string
from random import randint
from collections import Counter
from preprocess import preprocess
import json
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import normalize

vocabulary = []
business_list = {}
input_directory = "../json"
reviews = []
result = []
business_split = {}

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

    # # Remove hashtags
    # list_no_hashtags=[re.sub(r'#', '', i) for i in list_no_hyperlinks]

    # Remove Punctuation and split 's, 't, 've with a space for filter
    list_no_punctuation=[re.sub(r'['+string.punctuation+']+', ' ', i) for i in list_no_hyperlinks]

    # Remove multiple whitespace
    new_sent = ' '.join(list_no_punctuation)
    # Remove any words with 2 or fewer letters
    filtered_list = tw_tknzr.tokenize(new_sent)
    list_filtered = [re.sub(r'^\w\w?$', '', i) for i in filtered_list]

    # filtered_sent =' '.join(list_filtered)
    # clean_sent=re.sub(r'\s\s+', ' ', filtered_sent)
    # #Remove any whitespace at the front of the sentence
    # clean_sent=clean_sent.lstrip(' ')
    return stem_tokens(list_filtered)
    # return list_filtered

i=0
for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        print("collecting: "+filename)
        input_file = open(input_directory+"/"+filename, "r")
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
            
#             result.append([business_list[line["business_id"]], line["useful"], line["review_count"]])
            # tokens = p.preprocess(text)
            tokens = tweet_clean(text)
            temp = ' '.join(tokens)
            clean_sent=re.sub(r'\s\s+', ' ', temp)
            clean_sent=clean_sent.lstrip(' ')
            business_split[business_id]["text"].append(clean_sent)
            vocabulary.extend(tokens)
        input_file.close()

# vectorizer = CountVectorizer()
# result = vectorizer.fit_transform(reviews)

# vocabulary = vectorizer.get_feature_names()
vocabulary = set(vocabulary)
# print(vocabulary)
print(len(vocabulary))
print("finish collecting vocabulary")

vectorizer = CountVectorizer(vocabulary=vocabulary)

test_set = None
train_set = None
for business, features in business_split.items():
    if business % 500 == 0:
        print("spliting on: "+str(business))
    labels = np.array(features["labels"])
    text = features["text"]
    random = randint(0,4)
    tokens = vectorizer.transform(text).toarray()
    normed_tokens = normalize(tokens, axis=1, norm='l1')
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

np.save("train.npy", np.array(train_set))
np.save("test.npy", np.array(test_set))
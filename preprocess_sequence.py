import numpy as np
import sys
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from random import randint

tw_tknzr=TweetTokenizer(strip_handles=True, reduce_len=True)
cache_english_stopwords=stopwords.words('english')
stemmer = PorterStemmer()

def stem_tokens(tokens):
    
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tweet_clean(text, vocabulary):
    text = text.lower()
    
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Remove tickers
    sent_no_tickers=re.sub(r'\$\w*','',text)
    
    temp_tw_list = tw_tknzr.tokenize(sent_no_tickers)

    # Remove stopwords
    list_no_stopwords=[i for i in temp_tw_list if i not in cache_english_stopwords]

    # Remove hyperlinks
    list_no_hyperlinks=[re.sub("\d+", "", i) for i in list_no_stopwords]

    list_no_punctuation=[re.sub(r'['+string.punctuation+']+', ' ', i) for i in list_no_hyperlinks]

    list_filtered = [re.sub(r'^\w\w?$', '', i) for i in list_no_punctuation]

    list_filtered = [i for i in list_filtered if i not in vocabulary]
    return stem_tokens(list_filtered)


def main(filtered_city_dir, cityname, seq_dir):
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)

    if not os.path.exists(seq_dir+"/"+cityname):
        os.makedirs(seq_dir+"/"+cityname)
    input_file = open(filtered_city_dir+"/review_"+cityname+".json", "r")
    data = json.loads(input_file.read())
    reviews_text = []
    reviews_label = []
    reviews_business_id = []
    business_list = {}
    vocabulary = json.load(open("vocab_not_included.txt"))
    i = 0
    max_len = 0
    for line in data:
        if line["useful"] < 3:
            continue
        text = line["text"]
        tokens = tweet_clean(text, vocabulary)
        max_len = max(len(tokens), max_len)
        temp = ' '.join(tokens)
        clean_sent=re.sub(r'\s\s+', ' ', temp)
        clean_sent=clean_sent.lstrip(' ')
        reviews_text.append(clean_sent)
        if line["business_id"] not in business_list:
            business_list[line["business_id"]] = i
            i+=1
        reviews_business_id.append(business_list[line["business_id"]])
        reviews_label.append(line["useful"])
    print(max_len)

    max_num_words = 1000
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(reviews_text)
    sequences = tokenizer.texts_to_sequences(reviews_text)
    word_index = tokenizer.word_index
    max_seq_length = 0
    for i in range(len(sequences)):
        max_seq_length = max(max_seq_length, len(sequences[i]))
    print(max_seq_length)

    with open(seq_dir+"/"+cityname+"/word_index.json", "w") as write_file:
        json.dump(word_index, write_file)

    data = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    data.shape


    labels = np.array(reviews_label)

    labels.shape

    validation_spilit = 0.2

    business_ids_indices = np.argsort(np.array(reviews_business_id))
    data_sorted = data[business_ids_indices]
    business_ids_sorted = np.array(reviews_business_id)[business_ids_indices]
    labels_sorted = labels[business_ids_indices]
    _, unique_indices = np.unique(business_ids_sorted, return_index=True)
    indices_train = []
    indices_test = []
    for i in range(len(unique_indices)):
        start = unique_indices[i]
        end = len(business_ids_sorted)
        if (i+1 != len(unique_indices)):
            end = unique_indices[i+1]
        rand = randint(0,4)
        if rand == 0:
            for idx in range(start, end):
                indices_test.append(idx)
        else:
            for idx in range(start, end):
                indices_train.append(idx)
    indices_train = np.array(indices_train)
    indices_test = np.array(indices_test)
    print(len(indices_train))
    print(len(indices_test))
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_test)

    x_train = data_sorted[indices_train]
    y_train = labels_sorted[indices_train]
    x_val = data_sorted[indices_test]
    y_val = labels_sorted[indices_test]
    np.save(seq_dir+"/"+cityname+"/x_train.npy", x_train)
    np.save(seq_dir+"/"+cityname+"/y_train.npy", y_train)
    np.save(seq_dir+"/"+cityname+"/x_test.npy", x_val)
    np.save(seq_dir+"/"+cityname+"/y_test.npy", y_val)
    business_ids_eval = business_ids_sorted[indices_test]
    np.save(seq_dir+"/"+cityname+"/bid_test.npy", business_ids_eval)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
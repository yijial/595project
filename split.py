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

vocabulary = []
business_list = {}
input_directory = "../json"
reviews = []
result = []

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
			result.append([business_list[line["business_id"]], line["useful"], line["review_count"]])
			
			# tokens = p.preprocess(text)
			tokens = tweet_clean(text)
			vocabulary.extend(tokens)
			# for token in tokens: 
			# 	if token not in vocabulary:
			# 		vocabulary.append(token) # word dictionary vocabulary
			reviews.append(text) # array of reviews tokens
		input_file.close()

# vectorizer = CountVectorizer()
# result = vectorizer.fit_transform(reviews)

# vocabulary = vectorizer.get_feature_names()
vocabulary = set(vocabulary)
# print(vocabulary)
print(len(vocabulary))
print("finish collecting vocabulary")

vectorizer = CountVectorizer(vocabulary=vocabulary)
tokens = vectorizer.transform(reviews)
result_np = np.array(result)

business_split = {}
i=0
# result_np = np.array(result)
# result_np = np.column_stack(result_np, tokens)
for text in tokens:
# 	text = text.T
	# tokens = vectorizer.transform(text)
	# count = Counter(tokens)
	# temp = []
	# for word in vocabulary:
	# 	temp.append(count[word])
	num_words = np.sum(text)
	temp = text/num_words
	output=np.concatenate((result_np[i], temp.toarray()[0]))
	if result_np[i][0] not in business_split:
		business_split[result_np[i][0]] = []
	business_split[result_np[i][0]].append(output)
	i+=1

print("finish tokenization")

train_set = []
test_set = []
for business, reviews in business_split.items():
	if business % 500 == 0:
		print("spliting on: "+str(business))
	random = randint(0,4)
	if random == 0:
		test_set.extend(reviews)
	else:
		train_set.extend(reviews)

np.save("train.npy", np.array(train_set))
np.save("test.npy", np.array(test_set))


# business_split = {}
# i=0
# for text in reviews:
# 	# tokens = vectorizer.transform(text)
# 	# count = Counter(tokens)
# 	# temp = []
# 	# for word in vocabulary:
# 	# 	temp.append(count[word])
# 	num_words = sum(text)
# 	text = text/num_words
# 	result[i].extend(text)
# 	if result[i][0] not in business_split:
# 		business_split[result[i][0]] = []
# 	business_split[result[i][0]].append(result[i])
# 	i+=1

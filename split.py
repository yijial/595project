import sys
import os
import math
from random import randint
from collections import Counter
from preprocess import preprocess
import json
import numpy as np

vocabulary = []
business_list = {}
input_directory = "../json"
reviews = []
result = []
p = preprocess()

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
			
			tokens = p.preprocess(text)
			for token in tokens: 
				if token not in vocabulary:
					vocabulary.append(token) # word dictionary vocabulary
			reviews.append(tokens) # array of reviews tokens
		input_file.close()

print("finish collecting vocabulary")
bag_words = []
business_split = {}
i=0
for tokens in reviews:
	count = Counter(tokens)
	temp = []
	for word in vocabulary:
		temp.append(count[word])
	num_words = sum(temp)
	temp = np.array(temp)
	temp = temp/num_words
	result[i].extend(temp)
	if result[i][0] not in business_split:
		business_split[result[i][0]] = []
	business_split[result[i][0]].append(result[i])
	i+=1

print("finish tokenization")

train_set = []
test_set = []

for business, reviews in business_split.items():
	print("spliting on: "+str(business))
	random = randint(0,4)
	if random == 0:
		test_set.extend(reviews)
	else:
		train_set.extend(reviews)

np.save("train.npy", np.array(train_set))
np.save("test.npy", np.array(test_set))
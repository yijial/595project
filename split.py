import sys
import math
from collections import Counter
from preprocess import preprocess
import json
import numpy as np

vocabulary = []
input_file = open("../review_.json","r") # txt file input
reviews = []
result = []
business_list = {}
p = preprocess()
data = json.loads(input_file.read())
i=0
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

bag_words = []
nums_words = []

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
	nums_words.append(num_words)
	i+=1

result_np = np.array(result)
np.save("review_.npy", result_np)
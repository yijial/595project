from preprocess import preprocess

vocabulary = []
input_file = open("input_file","r") # txt file input
reviews = []
p = preprocess()
for line in input_file:
	tokens = p.preprocess(line)
	for token in tokens: 
		if token not in word_dict:
			vocabulary.append(token) # word dictionary vocabulary
	reviews.append(tokens) # array of reviews tokens
input_file.close()

bag_words = []
nums_words = []
for tokens in reviews:
	count = Counter(tokens)
	temp = []
	for word in vocabulary:
		temp.append(count[word])
	num_words = sum(temp)
	bag_words.append(temp/num_words)
	nums_words.append(num_words)

import re
import random
from PorterStemmer import PorterStemmer

class preprocess:
	stop_words = []
	file = open("stopwords", "r")
	for line in file:
		stop_words.append(line.rstrip())
	file.close()

	def removeSGML(self, in_text):
		return re.sub(r'<.*?>', '', in_text)

	"""
	helper function to randomize is / has / was
	with 1/3 probability
	stop words will not affect result list 
	"""
	def rand_s(self):
		s = ["is", "has", "was"]
		return random.choice(s)

	def rand_d(self):
		d = ["had", "would"]
		return random.choice(d) 

	def helper3(self, token):
		contraction1 = ["I", "you", "he", "she", "it", "we", "they", "that", "who", 
						"what", "where", "when", "why", "how"]
		# he's she's it's that's who's what's where's when's why's how's
		# contractions looks like possesives
		contraction2 = ["is", "are", "was", "were", "have", "has", "had", "wo", "would", 
						"do", "does", "did", "ca", "could", "should", "might", "must"]
		# 3. expand pocessives and contraction
		# xx's = is / was / has
		if token.find("'s") > 0:
			term = token.split("'s")[0]
			if term in contraction1: 
				return [term, self.rand_s()]
			if token == "let's":
				return ["let", "us"]
			else:
				return [term, "'s"]
		# 3. expand contractions
		# won't = will not, can't = cannot
		# xxn't = not
		# xx'll = will 
		# xx've = have
		# xx'd = would / had
		if token.find("n't") > 0:
			term = token.split("n't")[0]
			if term in contraction2:
				return [term, "not"]
			elif token == "won't":
				return ["will", "not"]
			elif token == "can't":
				return ["cannot"]
		cont = dict()
		cont["'ll"] = "will"
		cont["'ve"] = "have"
		cont["'re"] = "are" # "were" stemmed is "are",
		for key in cont:
			if token.find(key) > 0:
				term = token.split(key)[0]
				if term in contraction1:
					return [term, cont[key]]
		if token.find("'d") > 0:
			term = token.split()[0]
			if term in contraction1:
				return [term, self.rand_d]
		if token == "i'm":
			return ["i", "am"]
		return [token]

	"""
	tokenizeText: 
	1. split lines at spaces
	2. tokenize "./!," end point 
	3. tokenize "'"
	4. tokenize dates
	5. tokenize "-"
	"""
	def tokenizeText(self, in_text):
		tokens = []
		arr = in_text.rstrip().split(" ")
		length = len(arr)
		i = 0
		while i < length:
			token = arr[i]
			i += 1
			# 2. remove end points
			while len(token) > 0 and not token[len(token)-1].isalnum():
				token = token[:len(token)-1]

			# 3. contractions and possesives
			if token.find("'") > 0:
				tokens += self.helper3(token)
				continue

			# 4. US dates "april 1, 2018"
			months = ["january", "february", "march", "april", "may", "june", "july", 
					  "august", "september", "october", "november", "december"]
			abbrev_months = ["jan.", "feb.", "mar.", "apr.", "aug.", "sep.", "oct.", 
							 "nov.", "dec."]
			if (token in months or token in abbrev_months) and i < length:
				token2 = arr[i]
				if str(token2).isnumeric():
					token2 = int(re.search(r'\d+', token2).group())
					if token2 > 0 and token2 < 31:
						token += "/" + str(token2)
						i += 1
					if i < length:
						token3 = arr[i]
						if str(token3).isnumeric():
							token += "/" + token3
							i += 1

			# 5. keep words together
			token = ''.join(token.split("-"))

			if token != "":
				tokens.append(token)
		return tokens

	def removeStopwords(self, in_tokens):
		out_tokens = []
		for token in in_tokens:
			if token.lower() not in self.stop_words:
				out_tokens.append(token)
		return out_tokens

	def stemWords(self, in_tokens):
		stemmed_tokens = []
		p = PorterStemmer()
		for token in in_tokens:
			stemmed_tokens.append(p.stem(token, 0, len(token)-1))
		return stemmed_tokens

	def preprocess(self, content):
		content = self.removeSGML(content)
		tokens = self.tokenizeText(content)
		if tokens:
		#	tokens = self.removeStopwords(tokens)
			tokens = self.stemWords(tokens)
		return tokens
# Copyright 2019 Zhaozhen Liang
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet # for identiying the pos for lemmatizing

# problem remain to be solve:
# only do case-folding on the words that begin the sentence
# only those words that is not Name not Sepcial case

# nltk tag pos
tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}
AllposTag = [wordnet.VERB, wordnet.ADV,  wordnet.ADJ]

class Normalizer(object):
	"""handle normalization process"""
	def __init__(self):
		# initialize lemmatizer once at the beginning
		self.wordnetlemmatizer = WordNetLemmatizer()

	def lemmatize(self, tokens): 
		"""
		Input: a list of tokens/terms
		Return: a list of normalzied tokens
		"""
		return self.educatedBruteForceLemmatize(tokens)

	def educatedBruteForceLemmatize(self, tokens):
		new = []
		posTagList = self.getWordnetPos(tokens)
		for posTag in posTagList:
			# educated brute force method 
			if posTag[1][0].upper() in tag_dict:
				if tag_dict[posTag[1][0].upper()] == wordnet.VERB:
					# remove tense
					tmp = self.wordnetlemmatizer.lemmatize(posTag[0], 'v')
					new.append(tmp)
				else:
					# remove plural
					tmp = self.wordnetlemmatizer.lemmatize(posTag[0])
					# remove tense
					tmp = self.wordnetlemmatizer.lemmatize(tmp, 'v')
					new.append(tmp)
			else:
				# remove plural
				tmp = self.wordnetlemmatizer.lemmatize(posTag[0])
				# remove tense
				tmp = self.wordnetlemmatizer.lemmatize(tmp, 'v')
				new.append(tmp)
		self.caseFolding(new)
		return new
	
	def detectTypeAndLemmatize(self, tokens):
		new = []
		posTagList = self.getWordnetPos(tokens)
		for posTag in posTagList:
			if posTag[1][0].upper() in tag_dict:
				lemma = self.wordnetlemmatizer.lemmatize(posTag[0], tag_dict[posTag[1][0].upper()])
				if tag_dict[posTag[1][0].upper()] == wordnet.NOUN and lemma == posTag[0]:
					# try all pos tag lemmatize
					new.append(self.tryAllposTagLemmatize(posTag[0]))
				else:
					new.append(lemma)
			else:
				lemma = self.wordnetlemmatizer.lemmatize(posTag[0], wordnet.NOUN)
				if  lemma == posTag[0]:
					new.append(self.tryAllposTagLemmatize(posTag[0]))
				else:
					new.append(lemma)
		self.caseFolding(new)
		return new

	def tryAllposTagLemmatize(self, word):
		for i in AllposTag:
			lemma = self.wordnetlemmatizer.lemmatize(word, i)
			if lemma != word:
				return lemma
		return word

	def caseFolding(self, tokens):
		"""Convert all token in token list to lower case"""
		for index,token in enumerate(tokens):
			tokens[index] = token.lower()

	def getWordnetPos(self, words):
		"""return POS tag of a list of words"""
		return pos_tag(words)


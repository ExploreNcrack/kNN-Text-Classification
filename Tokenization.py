# Copyright 2019 Zhaozhen Liang
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

class Tokenizer:
	"""Class for handling tokenization process"""
	def __init__(self):
		# initialize a nltk tweettokenizer
		self.tweettokenizer = TweetTokenizer()
		self.english_stop_word_list = set(stopwords.words('english'))
	
	def tokenize(self, text):
		"""
		call this to handle tokenization process
		Input: string of text
		Return: a list of processed token
		"""
		# encode special meaningful charcter prevent tokenization will split it
		encodedText = self.encodeSpecialCharacterToPreserve(text)
		# tokenization
		tokens = self.tweettokenizer.tokenize(encodedText)
		# decode and replace back with the origin charcter
		self.decodePreserveSpecialCharacter(tokens)
		# get rid of "..." and "#"
		self.getRidNotMeaningfulCharacter(tokens)
		return tokens

	def getRidStopWord(self, tokens):
		"""
		Get rid of stop word in the list of tokens
		Input: takes in a list of tokens/terms
		Return: a list of tokens with stop word free
		"""
		stopWordfree = []
		for token in tokens:
			if token not in self.english_stop_word_list:
				stopWordfree.append(token)

		return stopWordfree


	def getRidNotMeaningfulCharacter(self, tokens):
		for index,token in enumerate(tokens):
			if token[0] == "#" and len(token) > 1:
				tokens[index] = token[1:].lower()
			elif token.find("...") == 0 and self.notAllpunctuation(token):
				tokens[index] = token[3:]


	def notAllpunctuation(self, s):
		for c in s:
			if c not in string.punctuation and (c.isalpha() or c.isdigit()):
				return True
		return False

	def containOnlyAlpha(self, s):
		for c in s:
			if c not in string.punctuation and not c.isalpha():
				return False
		return True

	def encodeSpecialCharacterToPreserve(self, text):
		wordList = text.split(" ")
		for index,word in enumerate(wordList):
			check = self.notAllpunctuation(word)
			if not check:
				continue
			if word.count(".") > 1 and self.containOnlyAlpha(word) and ".." not in word:
				if self.checkdot(word):
					wordList[index] = word.replace(".", "ztodp")
		return " ".join(wordList)

	def checkdot(self, word):
		dotSplit = word.split(".")
		for i in dotSplit:
			if self.numberOfAlphaContain(i) > 1:
				return False
		return True

	def numberOfAlphaContain(self, s):
		count = 0
		for i in s:
			if i.isalpha():
				count += 1
		return count 

	def decodePreserveSpecialCharacter(self, tokens):
		for index,token in enumerate(tokens):
			tokens[index] = token.replace("ztodp", ".")

	def allPunction(self, s):
		return not self.notAllpunctuation(s)
			
	def getRidPuncuation(self, tokens):
		"""
		Remove case "...." need to check if only contain punctuation
		Input: a list of tokens 
		Return: a list of tokens with only contain punctuation tokens filter
		"""
		newTokens = []
		for token in tokens:
			if self.notAllpunctuation(token):
				newTokens.append(token)
		return newTokens

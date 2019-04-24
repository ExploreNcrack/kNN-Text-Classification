#!/usr/bin/python3 
# set your own path

# execute the program by:
# q2/knn_classifier train.json test.json

"""
Goal of this program:
implements the kNN classification algorithm 
(Chapter 14) to classify the documents in the test file. 
Your program should take the two JSON files as input, 
and output the accuracy on the test set. 
Report accuracies for each class separately and for all classes together (in a separate line).
"""
import sys
import os
import json
import numpy as np
import math

def command_parser():
	if len(sys.argv) != 3:
		# number of argument is not correct
		print("Two arguments are needed:")
		print("1. the path of the training json file")
		print("2. the path of the test json file")
		print("% q2/knn_classifier train.json test.json")
		sys.exit(-1)

	trainJSONFile = sys.argv[1]
	testJSONFile = sys.argv[2]

	# check if the training json file exists
	if not os.path.isfile(trainJSONFile):
		print("The given training file does not exist")
		sys.exit(-1)
	# check if the test json file exists
	if not os.path.isfile(testJSONFile):
		print("The given test file does not exist")
		sys.exit(-1)
	# open files
	try:
		trainJSONFile = open(trainJSONFile,"r")
		testJSONFile = open(testJSONFile,"r")
	except IOError as error:
		print(error)
		sys.exit(-1)
	# load json data
	try:
		trainJSONData = json.load(trainJSONFile)
		testJSONData = json.load(testJSONFile)
	except Exception as error:
		print(error)
		sys.exit(-1)
	trainJSONFile.close()
	testJSONFile.close()
	return trainJSONData, testJSONData


def compute_tf_idf_weight(termFrequency, documentFrequency, number_of_document):
	if termFrequency == 0:
		return 0
	# tf * idf
	weight_of_term_in_document = (1+ math.log10(termFrequency)) * math.log10(number_of_document/documentFrequency)
	return weight_of_term_in_document


def compute_cosine_similarity(document_vector1, document_vector2):
	"""
	This function use two document tf-idf vector to compute the cosine angle between them
	and see how similar these two document are
	The higher the cosine score is, the smaller the angle between the two document vectos
	and thus more similar to each other they are
	"""
	document_vector1_len = np.sqrt( document_vector1.dot(document_vector1) )
	document_vector2_len = np.sqrt( document_vector2.dot(document_vector2) )
	cosine_of_angle_theta = np.dot(document_vector1,document_vector2) / (document_vector1_len * document_vector2_len)
	return cosine_of_angle_theta


def text_process(data_string):
	"""
	do cleaning text
		-removal of stop words
		-removal of Punctuation Characters 
		-stemming
	"""
	# perform tokenization
	tokens = tokenization.tokenize(data_string)
	# perform normalization
	tokens = normalization.lemmatize(tokens)
	# get rid of non-meaningful character after tokenization
	tokens = tokenization.getRidPuncuation(tokens)
	# get rid of stop word
	tokens = tokenization.getRidStopWord(tokens)
	return tokens

def clean_text(trainJSONData):
	"""
	This function will take in training data
	and do cleaning text
		-removal of stop words
		-removal of Punctuation Characters 
		-stemming
	and then compute the tf-idf 
	and assign it into the vector entry
	"""
	corpus = {}  # this will store all the term over all documents as key ; and the value will be [index,documentFrequency]
	documents_info = {}
	corpus_count = 0  # this also represent the index of each term in the corpus list ; also the dimension of the document vector space
	for document in trainJSONData:
		tokens = text_process(document["abstract"])
		documentLength = len(tokens)
		termFrequency = {} # store the term frequency for this document
		alreadyIncrement = {}
		for token in tokens:
			if token not in corpus:
				corpus_count += 1
				corpus[token] = [corpus_count,0]
			if token not in termFrequency:
				termFrequency[token] = 0
			if token not in alreadyIncrement:
				corpus[token][1] += 1
				alreadyIncrement[token] = None
			termFrequency[token] += 1
		# store stuff to the document info dictionary
		documents_info[document["title"]] = [tokens,document["type"],termFrequency,documentLength]

	# represent the documents as vector over corpus space dimension
	Document_vectors = []
	number_of_document = len(documents_info)
	# courpus_count will also be the dimension of the document vector space
	# for each document in the document_info
	for document_name,info in documents_info.items():
		# init dimension vector
		# each document vector entry will be the tf-idf weight of corresponding position term
		document_vector = np.zeros(corpus_count) # 1 x corpus_count vector
		terms_in_document = info[0]
		Class = info[1]
		termFrequency = info[2]
		documentLength = info[3]
		for term in terms_in_document:
			documentFrequency = corpus[term][1]
			index = corpus[term][0]
			# compute the tf-idf score and put it into the corresponding vector entriy
			document_vector[index-1] = compute_tf_idf_weight(termFrequency[term], documentFrequency, number_of_document)
		Document_vectors.append([document_name, document_vector, Class])

	return Document_vectors, corpus, number_of_document, corpus_count

def select_k():
	"""
	This function will select the 'best' k parameter
	for the classification
	"""

	return

def pre_processing(trainJSONData):
	"""
	This function will take in the training data in JSON format
	and first perform 
		cleaning text
	"""
	Document_vectors, corpus, number_of_document, corpus_count = clean_text(trainJSONData)
	#select_k()
	return Document_vectors, corpus, number_of_document, corpus_count

def find_kNN_determine_class(Document_vectors, test_document_vector, k):
	tmp_list_for_sort_similarity_score = []
	for index,document_vector in enumerate(Document_vectors):
		score = compute_cosine_similarity(document_vector[1],test_document_vector)
		tmp_list_for_sort_similarity_score.append([score,index])
	tmp_list_for_sort_similarity_score.sort(key=lambda dv:dv[0],reverse=True)
	classes_info = {}
	for i in range(k):
		index = tmp_list_for_sort_similarity_score[i][1]
		Class = Document_vectors[index][2]
		if Class not in classes_info:
			classes_info[Class] = 1
		else:
			classes_info[Class] += 1
	MAX = -1
	MAX_class = None
	for Class in classes_info:
		if classes_info[Class] > MAX:
			MAX = classes_info[Class]
			MAX_class = Class
	return MAX_class

def apply_kNN(test_document, Document_vectors, corpus, number_of_document, corpus_count, k):
	"""
	This function will take in one test document and
	return the label according to the k-neares-neighbours label
	"""
	# First we need to process the text document text
	tokens = text_process(test_document["abstract"])
	# find out the term freuqency of each term appear in the test document
	test_document_term_frequency = {}
	for token in tokens:
		if token not in test_document_term_frequency:
			test_document_term_frequency[token] = 1
		else:
			test_document_term_frequency[token] += 1
	# Then we represnt the test document as vector on the train document corpus dimension space
	test_document_vector = np.zeros(corpus_count) # 1 x corpus_count vector
	for token in tokens:
		if token in corpus:
			termFrequency = test_document_term_frequency[token]
			documentFrequency = corpus[token][1]
			index = corpus[token][0]
			test_document_vector[index-1] = compute_tf_idf_weight(termFrequency,documentFrequency,number_of_document)
	Class = find_kNN_determine_class(Document_vectors, test_document_vector, k)
	return Class

def apply_kNN_on_test_documents(testJSONData, Document_vectors, corpus, number_of_document, corpus_count, k):
	total_count = 0
	match_count = 0 
	for test_document in testJSONData:
		Class = apply_kNN(test_document, Document_vectors, corpus,  number_of_document, corpus_count, k)		
		if Class == test_document["type"]:
			match_count += 1
			print("Test document title: %-60s Predict type: %-13s Actual type: %-13s k:%d"%(test_document["title"],Class,test_document["type"],k))
		else:
			print("Test document title: %-60s Predict type: %-13s Actual type: %-13s k:%d -----wrong----*"%(test_document["title"],Class,test_document["type"],k))
		total_count += 1
	accuracy = match_count / total_count
	return accuracy

def main():
	"""
	The program must accept two command line arguments: 
	-train.json
	-test.json
	"""
	# first handle user input
	trainJSONData, testJSONData = command_parser()

	# import the text process after checking user input
	import Normalization 
	import Tokenization 

	# init text processing classes
	global normalization, tokenization 
	normalization = Normalization.Normalizer()
	tokenization = Tokenization.Tokenizer()

	print("Pre-processing begin >>>>>>>>")
	# Perform Data pre-processing (text processing and get each document terms)
	Document_vectors, corpus, number_of_document, corpus_count = pre_processing(trainJSONData)
	print("<<<<<<<< Pre-processing done")
	# apply the kNN
	best_accuary = -1
	best_k = -1
	decrease = 0
	k_parameter_accuracy = []
	# try all different parameter k
	# until if there are two consectively decreases 
	# then stop
	for k in range(1,number_of_document):
		print("Apply kNN begin with K=%d  >>>>>>>>"%(k))
		accuracy = apply_kNN_on_test_documents(testJSONData, Document_vectors, corpus, number_of_document, corpus_count, k)
		k_parameter_accuracy.append(accuracy)
		print("<<<<<<<< Apply kNN done with K=%d"%(k))
		print("Accuracy: "+str(accuracy)+"  with K=%d"%(k))
		if accuracy > best_accuary:
			best_accuary = accuracy
			best_k = k
		if k > 1 and accuracy < k_parameter_accuracy[k-2]:
			decrease += 1
		if decrease == 2:
			# if consectively decreasing break
			print("Two consectively decreasing accuracy! Stop here")
			break
	print("")
	print("Best Accuracy: %f  with parameter K=%d"%(best_accuary,best_k))


if __name__ == "__main__":
	main()
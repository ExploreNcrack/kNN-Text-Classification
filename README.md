# kNN classification

## Written by: Zhaozhen Liang (zhaozhen)

## Main Program: knn_classifier.py


## How to execute?
**python3 knn_classifier.py [train.json] [test.json]**


## Goal
The goal of this program is to first take in the training JSON data file (which contains numbers of documents, each document has "abstract","type","id","title") and proceed text processing on all of the documents in training data file and then represent each document as vector over the corpus dimension space with each vector entry as tf-idf weight (kind of like the idea of extracting the features from documents). This is the pre-processing step. After the pre-processing step, the program will then take all the information from training data file and apply it to the testing documents in testing data file. The program will also convert each testing document in the file into vector (over the same corpus dimension as the training document vector). And it will compute the cosine score between each testing document vector with all of the training documents. This represnts the angle between the vectors and also be an reference of how similar those documents are (the higher the cosine score, the smaller the angle is, and thus the more similar they are). In such way, we find the top k most similar training documents to each testing document and predict its type by the majority of the type of the top k most similar training documents.

## Program Design
### Text Process
For the text processing part, I use the classes that I implemented in Assignment1 to handle it.
</br>which includes:
* Tokenization 
* Normalization
These processes handles quite a lot of different cases, and preserves lots of meaningful words also maybe miss some edge cases.
</br>and I also added remove stop words from the documents. Since they are not as meaningful/informative words and appear very often.
### Parameter k
For the parameter k, of deciding how many top k of the nearest neighbour we should consider, I use the approach of parameter sweeping (brute force: try all different value) with a little bit of educated approximation. So I use a for loop to try all different k value start from 1 to the number of documents in the training file. And as I sweeping parameters I record down all the accuracy. If there are two consectively decreasing accuracy as I increases the parameter k, I stop the iterations(sweeping). This will prevent most of cases to try every different k values which is too costly and not necessary. But the best k parameter will not guarentee to be the best parameter k over all. Since it does not try all different k value. With a high chance that it will be a reasonable accuracy.
</br>With this testing data file, the best k parameter I tested on my own computer is k=5 with accuracy as 80%
</br>Accuracy can be affected by different version of NLTK.



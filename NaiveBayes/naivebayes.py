#!/usr/bin/python2.7

# python2 naivebayes.py Positive Negative Objective Neutral
# Basic classifiction functionality with Naive Bayes. 
# File originally provided for the assignment on classification (Information Retrieval course 2016/17)
# File changed for Information Retrieval Assignment
# File changed again for a project (Language Technology Project)
# Direct tweets towards the right category based on other already analyzed tweets
# Changes by Nathalie


import nltk.classify
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words
from classification import precision_recall

from random import shuffle
from os import listdir # to read files
from os.path import isfile, join # to read files
import sys

import string
import nltk

# import this library to go to the stem of the words
from nltk.stem import SnowballStemmer

# return all the filenames in a folder
def get_filenames_in_folder(folder):
	return [f for f in listdir(folder) if isfile(join(folder, f))]
	
def read_test_files(categories):
	test_feats = list ()
	print("\n##### Reading test files...")
	for category in categories:
		files = get_filenames_in_folder('TestSentences/' + category)
		num_files=0
		for f in files:
			datatest = open('TestSentences/' + category + '/' + f, 'r').read().decode("utf-8")
			
			tokenstest = word_tokenize(datatest)
			bag = bag_of_words(tokenstest)
			test_feats.append((bag, category))
			#print len(tokens)
			num_files+=1
		
		print ("  Category %s, %i testfiles read" % (category, num_files))

	return test_feats

# reads all the files that correspond to the input list of categories and puts their contents in bags of words
def read_files(categories):
	feats = list ()
	print("\n##### Reading files...")
	for category in categories:
		files = get_filenames_in_folder('Tweets/' + category)
		num_files=0
		for f in files:
			data = open('Tweets/' + category + '/' + f, 'r').read().decode("utf-8")
			
			# lowercase all words
			data = data.lower()
						
			# remove punctuation
			excludepunctuation = set(string.punctuation)
			datawithoutpunctuation="".join(word for word in data if word not in excludepunctuation)
								
			# filter out stopwords (100 most frequent words )
			# list of words from https://en.wikipedia.org/wiki/Most_common_words_in_English
			stopwords = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
			"it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
			"this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
			"or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
			"so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
			"when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
			"people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
			"than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
			"back", "after", "use", "two", "how", "our", "work", "first", "well", "even",
			"new", "want", "because", "any", "these", "give", "day", "most", "way", "us"]
			
			# filter out stopwords by putting them in a list, searching for the stopword and write them to a string
			dataliststopwords = datawithoutpunctuation.split()
			#dataliststopwords = data.split()
			datawithoutstopwordslist = []
			for word in dataliststopwords:
				if word not in stopwords:
					datawithoutstopwordslist.append(word)
			datawithoutstopwords= " ".join(datawithoutstopwordslist)
			
			# tokenize data
			datatokenized = nltk.word_tokenize(datawithoutstopwords)
			datatokenizedstring = " ".join(datatokenized)
			
			# create the stem of the data
		#	stemmer = SnowballStemmer("english")
		#	datastemmedinlist = []
			
			# stem all data		
		#	for word in datatokenized:
		#		stemofword = stemmer.stem(word)
		#		datastemmedinlist.append(stemofword)
			
		#	datastemmed = " ".join(datastemmedinlist) 
			
			tokens = word_tokenize(datatokenizedstring)
		#	tokens = word_tokenize(datastemmed)
			bag = bag_of_words(tokens)
			feats.append((bag, category))
			#print len(tokens)
			num_files+=1
		
		print ("  Category %s, %i files read" % (category, num_files))

	return feats



# splits a labelled dataset into two disjoint subsets train and test
def split_train_test(feats, split=0.9):
	train_feats = []
	dev_feats = []

	shuffle(feats) # randomise dataset before splitting into train and test
	cutoff = int(len(feats) * split)
	train_feats, dev_feats = feats[:cutoff], feats[cutoff:]	

	print("\n##### Splitting datasets...")
	print("  Training set: %i" % len(train_feats))
	print("  Dev set: %i" % len(dev_feats))
	return train_feats, dev_feats



# trains a classifier
def train(train_feats):
	# Naive Bayes classifier
	#nb_classifier = NaiveBayesClassifier.train(train_feats)
	classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)
	return classifier
	
	# La place classifier
	#from nltk.probability import LaplaceProbDist
	#classifier = nltk.classify.NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)
	#return classifier
	
	# DecisionTree classifier
	#classifier = nltk.classify.DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10)
	#return classifier


def calculate_f(precisions, recalls):
	f_measures = {}
	
	# calculate the F-scores for Positive
	fscorehigh = 2*(precisions['Positive']*recalls['Positive'])
	fscorelow = (precisions['Positive']+recalls['Positive'])
	fscore = fscorehigh/fscorelow
	f_measures['Positive'] = fscore
	
	# calculate the F-scores for Negative
	fscorehigh = 2*(precisions['Negative']*recalls['Negative'])
	fscorelow = (precisions['Negative']+recalls['Negative'])
	fscore = fscorehigh/fscorelow
	f_measures['Negative'] = fscore

	# calculcate the F-scores for Objective
	fscorehigh = 2*(precisions['Objective']*recalls['Objective'])
	fscorelow = (precisions['Objective']+recalls['Objective'])
	fscore = fscorehigh/fscorelow
	f_measures['Objective'] = fscore

	# calculcate the F-scores for Neutral
	fscorehigh = 2*(precisions['Neutral']*recalls['Neutral'])
	fscorelow = (precisions['Neutral']+recalls['Neutral'])
	fscore = fscorehigh/fscorelow
	f_measures['Neutral'] = fscore

	return f_measures



# prints accuracy, precision and recall
def evaluation(classifier, dev_feats, categories):
	print ("\n##### Evaluation...")
	print("  Accuracy: %f" % nltk.classify.accuracy(classifier, dev_feats))
	precisions, recalls = precision_recall(classifier, dev_feats)
	f_measures = calculate_f(precisions, recalls)  

	print(" |-----------|-----------|-----------|-----------|")
	print(" |%-11s|%-11s|%-11s|%-11s|" % ("category","precision","recall","F-measure"))
	print(" |-----------|-----------|-----------|-----------|")
	for category in categories:
		if precisions[category] is None:
			print("|%-11s|%-11s|%-11s|%-11s|" %(category, "NA","NA","NA"))
		else:
			print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category], f_measures[category]))
	print(" |-----------|-----------|-----------|-----------|")




# prints accuracy, precision and recall for the test data
def test_evaluation(classifier, test_feats, categories):
	print ("\n##### Evaluation...")
	precisions, recalls = precision_recall(classifier, test_feats)
	print("Precisions and recall scores of the test set")
	print("precisions", precisions)
	print("recalls", recalls)
	#print("\n")
	#f_measures = calculate_f(precisions, recalls)  

	#print(" |-----------|-----------|-----------|-----------|")
	#print(" |%-11s|%-11s|%-11s|%-11s|" % ("category","precision","recall","F-measure"))
	#print(" |-----------|-----------|-----------|-----------|")
	#for category in categories:
	#	if precisions[category] is None:
	#		print("|%-11s|%-11s|%-11s|%-11s|" %(category, "NA","NA","NA"))
	#	else:
			#f_measures = calculate_f(precisions, recalls)
			#print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category], f_measures[category]))
	#print(" |-----------|-----------|-----------|-----------|")
	print("\n")
	print(" The accuracy for the test set is: %f" % nltk.classify.accuracy(classifier, test_feats))



# show informative features
def analysis(classifier):
	print("\n##### Analysis...")
	classifier.show_most_informative_features(10)



# obtain the high information words
def high_information(feats, categories):
	print("\n##### Obtaining high information words...")

	labelled_words = [(category, []) for category in categories]

	#1. convert the formatting of our features to that required by high_information_words
	from collections import defaultdict
	words = defaultdict(list)
	all_words = list()
	for category in categories:
		words[category] = list()

	for feat in feats:
		category = feat[1]
		bag = feat[0]
		for w in bag.keys():
			words[category].append(w)
			all_words.append(w)
#		break

	labelled_words = [(category, words[category]) for category in categories]
	#print labelled_words

	#2. calculate high information words
	high_info_words = set(high_information_words(labelled_words))
	#print(high_info_words)
	print("  Number of words in the data: %i" % len(all_words))
	print("  Number of distinct words in the data: %i" % len(set(all_words)))
	print("  Number of distinct 'high-information' words in the data: %i" % len(high_info_words))

	return high_info_words


# read categories from arguments. e.g. "python2 naivebayes.py Positive Negative Objective Neutral"
categories = list()
for arg in sys.argv[1:]:
	categories.append(arg)



# main
test_feats = read_test_files(categories)
feats = read_files(categories)
high_info_words = high_information(feats, categories)
lijst = []

for N in range(10): # towards n-fold cross validation?
	train_feats, dev_feats = split_train_test(feats)
	classifier = train(train_feats)
	evaluation(classifier, dev_feats, categories)
	analysis(classifier)

	print(" ")
	# add the accuracyvalue to the list
	accuracydigit = nltk.classify.accuracy(classifier, dev_feats)
	lijst.append(accuracydigit)
		
	# print the items in the list
	print("The accuracy / accuracies for the dev set is / are: ")		
	for items in lijst:
		print(items)
		
	# check if there are 10 items inside the list	
	if len(lijst) == 10:
		total = 0
		# compute the total, and devide by 10, resulting in the average
		total = lijst[0] + lijst[1] + lijst[2] + lijst[3] + lijst[4] + lijst[5] + lijst[6] + lijst[7] + lijst[8] + lijst[9]
		#total = lijst[0] + lijst[1]
		average = 0
		average = total / 10
		print("\nThe average accuracy for the dev set is: ")
		print(average)
		
		test_evaluation(classifier, test_feats, categories)
		


	



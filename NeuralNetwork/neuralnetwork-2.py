# Run program with; python neuralnetwork.py
# A simple neural network in categorize sentences with figures of speech into the corresponding sentiment based on precategorized tweets
# Language Technology Project 2017
# Nathalie

# Import libraries
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from random import shuffle

print("\n")

# Open tweets and concatenate them
positive_sentences = [l.strip() for l in open("TestSentences/Positive/newpositivetweets.txt").readlines()]
negative_sentences = [l.strip() for l in open("TestSentences/Negative/newnegativetweets.txt").readlines()]
objective_sentences = [l.strip() for l in open("TestSentences/Objective/newobjectivetweets.txt").readlines()]
neutral_sentences = [l.strip() for l in open("TestSentences/Neutral/newneutraltweets.txt").readlines()]
#sentences = np.concatenate([positive_sentences,negative_sentences, objective_sentences, neutral_sentences], axis=0)
sentences = np.concatenate([positive_sentences,negative_sentences, objective_sentences, neutral_sentences])

# split data on train and development data
split = 0.9	
shuffle(sentences)
cutoff = int(len(sentences) * split)
X_train, X_dev = sentences[:cutoff], sentences[cutoff:]

listofytrain = []
for line in X_train:
	sentimentofline =  line.split()
	listofytrain.append(sentimentofline[0])

y_train = listofytrain

listofydev = []
for line in X_dev:
	sentimentofline2 = line.split()
	listofydev.append(sentimentofline2[0])
	
y_dev = listofydev
	


# Open corresponding sentiments of the tweets and concatenate them
#positive_sentiment = [l.strip() for l in open("TestSentences/Positive/amountofpositives.txt").readlines()]
#negative_sentiment = [l.strip() for l in open("TestSentences/Negative/amountofnegatives.txt").readlines()]
#objective_sentiment = [l.strip() for l in open("TestSentences/Objective/amountofobjectives.txt").readlines()]
#neutral_sentiment = [l.strip() for l in open("TestSentences/Neutral/amountofneutrals.txt").readlines()]
#sentimentofsentences = np.concatenate([positive_sentiment, negative_sentiment, objective_sentiment, neutral_sentiment])

# Open the sentences with figures of speech and concatenate them
positive_test_sentences = [l.strip() for l in open("TestData/Positive/positivetweets.txt").readlines()]
negative_test_sentences = [l.strip() for l in open("TestData/Negative/negativetweets.txt").readlines()]
objective_test_sentences = [l.strip() for l in open("TestData/Objective/objectivetweets.txt").readlines()]
neutral_test_sentences = [l.strip() for l in open("TestData/Neutral/neutraltweets.txt").readlines()]
test_sentences = np.concatenate([positive_test_sentences, negative_test_sentences, objective_test_sentences, neutral_test_sentences])

# Open the corresponding sentiments of the sentences with figures of speech and concatenate them
positive_test_sentiment = [l.strip() for l in open("TestData/Positive/amountofpositivestest.txt").readlines()]
negative_test_sentiment = [l.strip() for l in open("TestData/Negative/amountofnegativestest.txt").readlines()]
objective_test_sentiment = [l.strip() for l in open("TestData/Objective/amountofobjectivestest.txt").readlines()]
neutral_test_sentiment = [l.strip() for l in open("TestData/Neutral/amountofneutralstest.txt").readlines()]
sentimentoftestsentences = np.concatenate([positive_test_sentiment, negative_test_sentiment, objective_test_sentiment, neutral_test_sentiment])

# Assign the tweets and their sentiments to the train & development set
# Assign the sentences and their sentiments to the test set
#X_train = sentences
#y_train = sentimentofsentences
#X_dev = sentences
#y_dev = sentimentofsentences
X_test = test_sentences
y_test = sentimentoftestsentences

print(len(X_train), " items as training data")
print(len(y_train), " sentimentitems as training data")
print("")
print(len(X_dev), " items as development data")
print(len(y_dev), " sentimentitems as development data")
print("")
print(len(X_test), " items as test data")
print(len(y_test), " sentimentitems as testdata")
print("")

#print(y_test)

# Map all words to indices, then create n-hot vector
from collections import defaultdict

w2i = defaultdict(lambda: len(w2i))
PAD = w2i["<pad>"]
UNK = w2i["<unk>"]
t2i = defaultdict(lambda: len(t2i))
TPAD = w2i["<pad>"]

X_train_num = [[w2i[word] for word in sentence] for sentence in X_train]
w2i = defaultdict(lambda: UNK, w2i) # freeze
X_dev_num = [[w2i[word] for word in sentence] for sentence in X_dev]
X_test_num = [[w2i[word] for word in sentence] for sentence in X_test]

y_train_num=[[t2i[tag] for tag in sentence] for sentence in y_train]
t2i = defaultdict(lambda: UNK, t2i)
y_dev_num = [[t2i[tag] for tag in sentence] for sentence in y_dev]
y_test_num = [[t2i[tag] for tag in sentence] for sentence in y_test]

np.unique([y for sent in y_train for y in sent])
num_classes = len(np.unique([y for sent in y_train for y in sent]))

num_labels = len(np.unique([y for sent in y_train for y in sent]))
y_train_1hot = [np_utils.to_categorical([t2i[tag] for tag in instance_labels], num_classes = num_labels) for instance_labels in y_train]
y_dev_1hot = [np_utils.to_categorical([t2i[tag] for tag in instance_labels], num_classes= num_labels) for instance_labels in y_dev]
y_test_1hot = [np_utils.to_categorical([t2i[tag] for tag in instance_labels], num_classes= num_labels) for instance_labels in y_test]

max_sentence_length=max([len(s.split(" ")) for s in X_train] + [len(s.split(" ")) for s in X_dev] + [len(s.split(" ")) for s in X_test])

print("This step takes pretty long")

from keras.preprocessing import sequence

vocab_size = len(w2i)
embeds_size= 64
X_train_pad = sequence.pad_sequences(X_train_num, maxlen=max_sentence_length, value=PAD)
X_dev_pad = sequence.pad_sequences(X_dev_num, maxlen=max_sentence_length, value=PAD)
X_test_pad = sequence.pad_sequences(X_test_num, maxlen=max_sentence_length, value=PAD)
y_train_1hot_pad = sequence.pad_sequences(y_train_1hot, maxlen = max_sentence_length, value = TPAD)
y_dev_1hot_pad = sequence.pad_sequences(y_dev_1hot, maxlen = max_sentence_length, value = TPAD)
y_test_1hot_pad = sequence.pad_sequences(y_test_1hot, maxlen = max_sentence_length, value = TPAD)

np.random.seed(113) #set seed before any keras import
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed

# Changes to the model can be made in here
model = Sequential()
model.add(Embedding(vocab_size, embeds_size, input_length=max_sentence_length, mask_zero=True))
model.add(LSTM(40, return_sequences=True))
model.add(TimeDistributed(Dense(num_labels)))
model.add(Activation('softmax'))
#model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
#model.add(Activation('linear'))
#model.add(Activation('hard_sigmoid'))
#model.add(Activation('tanh'))
#model.add(Activation('softsign'))
#model.add(Activation('softplus'))
#model.add(Activation('elu'))
model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

# The actual fitting of the data takes place here
model.fit(X_train_pad, y_train_1hot_pad, epochs=11)

# Calculate the accuracy for the Development set
loss, accuracy = model.evaluate(X_dev_pad, y_dev_1hot_pad)
print("\n Accuracy of the dev set: ", accuracy *100)

# Calculate the accuracy for the Test set
loss, accuracy = model.evaluate(X_test_pad, y_test_1hot_pad)
print("\n Accuracy of the test set: ", accuracy *100)

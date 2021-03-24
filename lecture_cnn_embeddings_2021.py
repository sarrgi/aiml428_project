# -*- coding: utf-8 -*-
"""lecture-CNN-embeddings-2021.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kmvRyc9O-cHRXGpvYkG74eT5vWflOUxv
"""

import pandas
url='https://ecs.wgtn.ac.nz/foswiki/pub/Courses/AIML428_2021T1/LectureSchedule/simple-review.csv'
data = pandas.read_csv(url, encoding="latin-1")
data

#To load the data file this way takes time.
#You may use github to save the file and then load, or use Google Drive

from sklearn import model_selection, preprocessing, metrics
# split the dataset into training and validation datasets
train_x, test_x, train_y, test_y = model_selection.train_test_split(data['text'], data['label'], shuffle=False)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
print(train_x[0:10])
print(train_y[0:10])

# word to integer IDs, documents padded to the same length

import numpy
from keras.preprocessing import text, sequence

# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(data['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=70)
train_seq_x[0]



# CNN parameters from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb
# do not use pre-trained embedding
# Train word embedding as part of the model
# the embedding weights are randomly initialised

from keras.models import Sequential
from keras import layers

vocab_size = len(word_index)+1

model = Sequential()
model.add(layers.Embedding(vocab_size, 50))
model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))

model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_seq_x,
                    train_y,
                    epochs=10,
                    batch_size=512,
                    validation_data=(test_seq_x, test_y),
                    verbose=1)
results = model.evaluate(test_seq_x, test_y)

print(results)

#The glove file is big. To load it this way takes a long time; To upload from local file would take even longer
#You should use Google Drive

# the pre-trained word-embedding is downloaded from
#http://nlp.stanford.edu/data/glove.6B.zip


# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove*.zip
# !ls
# !pwd

import numpy

def load_embedding(filename):
    # load embedding into memory
    file = open(filename,'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] =numpy.asarray(parts[1:], dtype='float32')
    return embedding

# load embedding from file
raw_embedding = load_embedding('glove.6B.50d.txt')
raw_embedding



# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = numpy.zeros((vocab_size, 50))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        if word in embedding.keys():
            weight_matrix[i] = embedding.get(word)
    return weight_matrix

# get vectors in the right order
word_vectors = get_weight_matrix(raw_embedding, token.word_index)

print(word_index['book'])
print(word_vectors[14])

# This count the non-zeros in the word_vectors.
#This is important to make sure you have used the pretrained embedding correctly
vocab_size = len(word_index)+1
nonzero_elements = numpy.count_nonzero(numpy.count_nonzero(word_vectors, axis=1))
nonzero_elements / vocab_size

from keras.models import Sequential
from keras import layers

vocab_size=len(word_index)+1
maxlen=70
embedding_dim=50

# define model
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[word_vectors],
                           input_length=maxlen,
                           trainable=False))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(train_seq_x, train_y, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(test_seq_x, test_y, verbose=0)
print('Test Accuracy: %f' % (acc*100))

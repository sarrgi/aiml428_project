import time
import numpy
import pandas
from keras.preprocessing import text, sequence
from sklearn import model_selection, preprocessing, metrics
from keras.models import Sequential
from keras import layers

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



def read_data(url):
    data = pandas.read_csv(url, encoding="latin-1")
    return data


def construct_model(vocab_size, embedding_dim, word_vectors, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size,
                               embedding_dim,
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

    return model


def train_model(model, train_seq_x, test_seq_x, train_y, test_y):
    # fit network
    model.fit(train_seq_x, train_y, epochs=10, verbose=2)

    # evaluate
    loss, acc = model.evaluate(test_seq_x, test_y, verbose=0)
    print('Test Accuracy: %f' % (acc*100))


if __name__ == "__main__":
    start_time = time.time()

    # read text data
    data = read_data('https://ecs.wgtn.ac.nz/foswiki/pub/Courses/AIML428_2021T1/LectureSchedule/simple-review.csv')
    # print(data)

    # load pretrained glove model
    raw_embedding = load_embedding('glove/glove.6B.50d.txt')

    # split the dataset into training and validation datasets
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data['text'], data['label'], shuffle=False)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    print(train_x[0:10])


    # set up token
    token = text.Tokenizer()
    token.fit_on_texts(data['text'])
    word_index = token.word_index

    # text to sequence of tokens, padded to ensure equal length vecs
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=70)


    # create word vecs
    word_vectors = get_weight_matrix(raw_embedding, token.word_index)

    # print(word_index['book'])
    # print(word_vectors[14])

    # vocab_size = len(word_index)+1
    # nonzero_elements = numpy.count_nonzero(numpy.count_nonzero(word_vectors, axis=1))
    # print("?", nonzero_elements / vocab_size)

    vocab_size=len(word_index)+1
    maxlen=70
    embedding_dim=50

    model = construct_model(vocab_size, embedding_dim, word_vectors, maxlen)
    train_model(model, train_seq_x, test_seq_x, train_y, test_y)



    print("Full time take: %s seconds." % round((time.time() - start_time), 2))

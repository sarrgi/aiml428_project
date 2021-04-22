import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import xml.etree.ElementTree as et
import re
import math
from itertools import chain
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import string
try:
    import cPickle as pickle
except:
    import pickle

from tensorflow.keras import backend as K


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


global_word_vecs = "-1"
def set_global_word_vecs(val):
    """
    Nasty fix to work around could not clone error when passing word_vecs.
    """
    global global_word_vecs
    global_word_vecs = val



def create_embedding_matrix(filepath, word_index, embedding_dim, vocab_size):
    """
    Create the embedding matrix using the pre-trained glove file.
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def get_df(filepath_dict):
    """
    Store the datasets into a pandas dataframe.
    """
    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    df = pd.concat(df_list)
    return df


def plot_history(history):
    """
    Plot the models performance in terms of loss and accuracy.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    #
    plt.show()


def create_model(num_filters, kernel_size, hidden, vocab_size, embedding_dim, sentence_len, word_vecs):
    """
    Define the model architecture here.
    """
    model = Sequential()
    model.add(layers.Embedding(
                                vocab_size,
                                embedding_dim,
                                input_length=sentence_len,
                                weights=[word_vecs],
                                trainable=True))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(hidden, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def create_model_hyper_tuning(num_filters, kernel_size, hidden, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(
        vocab_size,
        embedding_dim,
        input_length=maxlen,
        weights=[global_word_vecs],
        trainable=True))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(hidden, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model in terms of classification accuracy on the test and training sets.
    """
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


def calculate_nonzero(embedding_matrix, vocab_size):
    """
    Calulate the percentage of words which are non-zero (are stored in the pretrained corpus).
    """
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    return nonzero_elements / vocab_size


def parse_file_name(str):
    """
    Removes excess path characters from file path, so that you are left with just the file name.
    """
    name = re.search("[a-zA-Z0-9]*\.xml", str).group(0)
    removed_extension = name[:-4]
    return removed_extension


def parse_truth_table(location):
    """
    Parse the txt truth table into a list of usable dicts.
    """
    f = open(location, "r")
    lines = f.readlines()

    dicts = []
    for line in lines:
        split = line.split(":::")
        d = {}
        d['file'] = split[0]
        d['target'] = split[1]
        d['sub_class'] = split[2][:-1] #remove \n character

        dicts.append(d)

    return dicts


def read_data(location):
    """
    Read in the data from an xml file.
    """
    all_tweets = []
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r", encoding="utf8") as file:
            # set first object in list to filename
            tweets = [parse_file_name(filename)]

            # parse xml document into list
            root = et.parse(file).getroot()
            for documents in root:
                for tweet in documents:
                    text = tweet.text
                    tweets.append(text)

            # append all files list
            all_tweets.append(tweets)
    return all_tweets


def create_targets(input, truth_table):
    """
    Create targets for an input array based on it's respective truth table.
    male = 0
    female = 1
    bot = 2
    """
    targs = []

    for i in input:
        file = i[0]
        targ = "not found"
        for t in truth_table:
            # find corresponding target
            if t['file'] == file:
                targ = t['target']
                # get sub target
                if targ == "human":
                    targ  = t['sub_class']

        # check target found and store if so
        if targ == "not found":
            print(file)
            raise create_targets("err", "err")
        else:
            if targ == "male":
                targs.append([1,0,0])
            elif targ == "female":
                targs.append([0,1,0])
            elif targ == "bot":
                targs.append([0,0,1])

    return targs


def flatten_input(input):
    """
    Flatten 2D input array into singluar array.
    """
    for i in range(len(input)):
        input[i] = input[i][1:]

    return list(chain.from_iterable(input))


def flatten_targets(targets, repeat_len):
    """
    Flatten targets into a 1d array.
    """
    size = len(targets) * repeat_len
    flat = np.zeros((size, 3), dtype=np.int64)

    idx = 0
    for t in targets:
        for i in range(repeat_len):
            flat[idx] = t
            idx += 1

    return flat


def get_longest_input(input_arr):
    """
    Get length of longest tweet in input array.
    """
    max = -1
    for t in input_arr:
        for tt in t:
            if len(tt) > max:
                max = len(tt)

    return max


def to_lower(input):
    """Convert a 2d input array to all lower case."""
    for i in range(len(input)):
        input[i] = input[i].lower()

    return input


def remove_urls(input):
    """
    Convert all instances of urls to the word url.
    """
    for i in range(len(input)):
        input[i] = re.sub(r"http\S+", "url", input[i])

    return input


def convert_mentions(input):
    """
    Convert all username mentions.
    """
    for i in range(len(input)):
        input[i] = re.sub(r"@[a-zA-Z0-9\_]+", "username", input[i])

    return input


def remove_stopwords(input):
    """
    Remove all stop words from input.
    """
    stop_words = set(stopwords.words('english'))

    for i in range(len(input)):
        # remove stop words
        word_tokens = word_tokenize(input[i])
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        # convert back to single sentence
        complete_sentence = " ".join(filtered_sentence)
        # store
        input[i] = complete_sentence

    return input


def remove_punctuation(input):
    """
    Remove all punctuation from input.
    """
    for i in range(len(input)):
        input[i] = re.sub(r"[^\w\s]", "", input[i])

    return input

if __name__ == "__main__":
    # required downloads to make nltk stopwords library work
    nltk.download('punkt')
    nltk.download('stopwords')

    # ////////////////////////////////////////////// read in and store data //////////////////////////////////////////////

    # gpu log check
    print(tf.config.list_physical_devices('GPU'))
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    print(tf.test.is_built_with_cuda())

    # read in data
    test_en = read_data("data/pandata/test/en/*.xml")
    train_en = read_data("data/pandata/train/en/*.xml")

    # read in truth tables
    en_test_truth = parse_truth_table("data/pandata/truth-tables/en-test.txt")
    en_train_truth =  parse_truth_table("data/pandata/truth-tables/en-truth.txt")

    # create targets array
    test_en_targets = create_targets(test_en, en_test_truth)
    train_en_targets = create_targets(train_en, en_train_truth)

    # flatten input
    test_en_input = flatten_input(test_en)
    train_en_input = flatten_input(train_en)

    # flatten targets
    test_en_targets = flatten_targets(test_en_targets, len(test_en[0]))
    train_en_targets = flatten_targets(train_en_targets, len(train_en[0]))

    # randomly extract a small part of dataset for testing (too expensive to run the whole dataset)
    train_en_input, _, train_en_targets, _ = model_selection.train_test_split(train_en_input, train_en_targets, train_size=0.1, shuffle=True)
    _, test_en_input, _, test_en_targets = model_selection.train_test_split(test_en_input, test_en_targets, test_size=0.1, shuffle=True)


    # ////////////////////////////////////////////// pre processing //////////////////////////////////////////////

    # set all tweets to lower case
    test_en_input = to_lower(test_en_input)
    train_en_input = to_lower(train_en_input)

    # remove urls
    test_en_input = remove_urls(test_en_input)
    train_en_input = remove_urls(train_en_input)
    # exit(1)

    # remove mentions
    test_en_input = convert_mentions(test_en_input)
    train_en_input = convert_mentions(train_en_input)

    # remove stop words
    test_en_input = remove_stopwords(test_en_input)
    train_en_input = remove_stopwords(train_en_input)

    # remove punctuation
    test_en_input = remove_punctuation(test_en_input)
    train_en_input = remove_punctuation(train_en_input)

    # remove/convert emojis



    # ////////////////////////////////////////////// get input ready for model //////////////////////////////////////////////

    # get longest tweet
    max_tweet = get_longest_input(test_en + train_en)
    # set length to pad sentences to (zeros at end of vector) (round to next hundred)
    sentence_len = int(math.ceil(max_tweet / 100.0)) * 100

    # set embedding dim size (must match glove file...)
    embedding_dim = 50

    # create tokenizer (note: num_words specifies the top n words to keep)
    tokenizer = Tokenizer(num_words=10000) #28987
    tokenizer.fit_on_texts(train_en_input)

    # convert sentences to integers (tokens)
    train_en_input = tokenizer.texts_to_sequences(train_en_input)
    test_en_input = tokenizer.texts_to_sequences(test_en_input)

    # pad sentences so theyre all the same length
    train_en_input = pad_sequences(train_en_input, padding='post', maxlen=sentence_len)
    test_en_input = pad_sequences(test_en_input, padding='post', maxlen=sentence_len)

    # store length of vocab (for model params and embedding matrix calcs)
    vocab_size = len(tokenizer.word_index) + 1

    # load the pretrained glove model into a matrix (file stored locally)
    # r"D:\UNI\Fourth Year\AIML428\glove.6B\glove.6B.50d.txt"
    embedding_matrix = create_embedding_matrix("glove/full_corp_min_2.txt",
                                               tokenizer.word_index,
                                               embedding_dim,
                                               vocab_size)

    # nasty fix to work around could not clone error when passing word_vecs
    set_global_word_vecs(embedding_matrix)
    #
    # Calculate the amount of words covered by GloVe
    print("Percent of vocabulary covered by GloVe:", calculate_nonzero(embedding_matrix, vocab_size))
    #

    # ////////////////////////////////////////////// hyper paramter tuning //////////////////////////////////////////////

    # Parameter grid for grid search
    param_grid = dict(num_filters=[32, 64, 128, 256],
                      kernel_size=[3, 5, 7],
                      hidden=[10, 20, 30, 40, 50],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[sentence_len],
                      )

    model = KerasClassifier(build_fn=create_model_hyper_tuning,
                            epochs=10,
                            batch_size=10,
                            verbose=False)


    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=1)

    grid_result = grid.fit(train_en_input, train_en_targets)

    test_accuracy = grid.score(test_en_input, test_en_targets)

    # get values back out of grid
    vocab_size = grid_result.best_params_['vocab_size']
    num_filters = grid_result.best_params_['num_filters']
    kernel_size = grid_result.best_params_['kernel_size']
    hidden = grid_result.best_params_['hidden']

    print("Best vocab_size:", vocab_size)
    print("Best num_filters:", num_filters)
    print("Best kernel_size:", kernel_size)
    print("Best hidden_size:", hidden)

    # ////////////////////////////////////////////// run and evaluate model //////////////////////////////////////////////

    # Create the model using best params
    model = create_model(num_filters, kernel_size, hidden, vocab_size, embedding_dim, sentence_len, embedding_matrix)

    # summarize model architecture
    model.summary()

    # train the model
    history = model.fit(train_en_input, train_en_targets, epochs=30, verbose=True, validation_split = 0.2, batch_size=10)

    # evaluate model
    evaluate_model(model, train_en_input, train_en_targets, test_en_input, test_en_targets)
    # plot_history(history)

    exit(1)

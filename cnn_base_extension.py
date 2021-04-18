import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session
from keras.models import Sequential
from keras import layers

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


def create_model(vocab_size, embedding_dim, setence_len, word_vecs):
    """
    Define the model architecture here.
    Currently using the structure from: https://realpython.com/python-keras-text-classification/.
    """
    model = Sequential()
    model.add(layers.Embedding(
                                vocab_size,
                                embedding_dim,
                                input_length=setence_len,
                                weights=[word_vecs],
                                trainable=True))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
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


def read_data(location):
    for filename in glob.glob(location):
        print(filename)




if __name__ == "__main__":


    test_en = read_data("data/pandata/test/en/*.xml")













    # original baseline below
    exit(1)


    # datset locations
    filepath_dict = {'yelp':   'data/yelp_labelled.txt',
                     'amazon': 'data/amazon_cells_labelled.txt',
                     'imdb':   'data/imdb_labelled.txt'}

    # store dataset in pandas dataframes
    df = get_df(filepath_dict)

    # run through model for each dataset
    for source in df['source'].unique():
        # clear weights from previous keras models
        clear_session()

        # load data (sentences and labels)
        df_source = df[df['source'] == source]
        sentences = df_source['sentence'].values
        y = df_source['label'].values

        # split dataset into test and train
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

        # set embedding dim size (must match glove file...)
        embedding_dim = 50
        # set length to pad sentences to (zeros at end of vector)
        setence_len = 100

        # create tokenizer (note: num_words specifies the top n words to keep)
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(sentences_train)

        # convert sentences to integers (tokens)
        X_train = tokenizer.texts_to_sequences(sentences_train)
        X_test = tokenizer.texts_to_sequences(sentences_test)

        # pad sentences so theyre all the same length
        X_train = pad_sequences(X_train, padding='post', maxlen=setence_len)
        X_test = pad_sequences(X_test, padding='post', maxlen=setence_len)

        # store length of vocab (for model params and embedding matrix calcs)
        vocab_size = len(tokenizer.word_index) + 1

        # load the pretrained glove model into a matrix (file stored locally)
        # r"D:\UNI\Fourth Year\AIML428\glove.6B\glove.6B.50d.txt"
        embedding_matrix = create_embedding_matrix("glove/glove.6B.50d.txt",
                                                   tokenizer.word_index,
                                                   embedding_dim,
                                                   vocab_size)

        # Calculate the amount of words covered by GloVe
        print("Percent of vocabulary covered by GloVe:", calculate_nonzero(embedding_matrix, vocab_size))

        # Create the model
        model = create_model(vocab_size, embedding_dim, setence_len, embedding_matrix)

        # summarize model architecture
        model.summary()

        # train the model
        history = model.fit(X_train, y_train, epochs=10, verbose=False, validation_data=(X_test, y_test), batch_size=10)

        # evaluate model
        evaluate_model(model, X_train, y_train, X_test, y_test)
        plot_history(history)

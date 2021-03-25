import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from keras.backend import clear_session
from keras.models import Sequential
from keras import layers

def lr(df):
    for source in df['source'].unique():
        df_source = df[df['source'] == source]
        sentences = df_source['sentence'].values
        y = df_source['label'].values

        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test  = vectorizer.transform(sentences_test)

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print('Accuracy for {} data: {:.4f}'.format(source, score))


def get_df(filepath_dict):
    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    df = pd.concat(df_list)
    return df



if __name__ == "__main__":
    filepath_dict = {'yelp':   'data/yelp_labelled.txt',
                     'amazon': 'data/amazon_cells_labelled.txt',
                     'imdb':   'data/imdb_labelled.txt'}

    df = get_df(filepath_dict)

    # lr(df)


    # keras below

    for source in df['source'].unique():
        df_source = df[df['source'] == source]
        sentences = df_source['sentence'].values
        y = df_source['label'].values


        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test  = vectorizer.transform(sentences_test)

        input_dim = X_train.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()

        history = model.fit(X_train, y_train,epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=10)

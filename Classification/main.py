import os
import pickle
import random

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dense, Input, LSTM, GRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import optimizers

maxlen = 200
max_words = 20000

categories = ['business', 'entertainment', 'politics', 'sport', 'tech']


def main():
    # Read BBC news data and get texts and labels
    # Each row of texts and labels corresponds to a single file
    # Total of 2225 files
    news = pd.read_json("News_Category_Dataset_v2.json", lines=True)
    news["text"] = news["headline"] + " " + news["short_description"]
    news = news.drop(["authors", "date", "link", "headline", "short_description"], axis=1)
    news = news.loc[
        (news['category'] != "WORLDPOST") & (news['category'] != "PARENTING") & (news['category'] != "BLACK VOICES") & (
                news['category'] != "LATINO VOICES") & (news['category'] != "QUEER VOICES") & (
                news['category'] != "WEIRD VOICES") & (news['category'] != "IMPACT") & (
                news['category'] != "GOOD NEWS") & (news['category'] != "TASTE") & (
                news['category'] != "HOME & LIVING") & (news['category'] != "PARENTS") & (
                news['category'] != "PARENTING") & (news['category'] != "WOMEN") & (news['category'] != "DIVORCE") & (
                news['category'] != "WEIRD NEWS") & (news['category'] != "FIFTY") & (
                news['category'] != "THE WORLDPOST") & (news['category'] != "GREEN") & (
                news['category'] != "WEDDINGS") & (news['category'] != "ARTS") & (
                news['category'] != "CULTURE & ARTS") & (news['category'] != "STYLE") & (news['category'] != "COLLEGE")]
    news["category"] = news["category"].astype('category')
    news["category_code"] = news["category"].cat.codes
    news = news.reset_index(drop=True)
    dictionary = dict(enumerate(news["category"].cat.categories))
    save_dictionary(dictionary)
    # data_dir = 'bbc'
    # labels = []
    # texts = []
    # label_count = 0
    # for label_type in categories:
    #     dir_name = os.path.join(data_dir, label_type)
    #     for fname in os.listdir(dir_name):
    #         f = open(os.path.join(dir_name, fname), encoding="utf8", errors='ignore')
    #         texts.append(f.read())
    #         f.close()
    #         labels.append(label_count)
    #     label_count = label_count + 1
    index_for_validation = []
    for c in range(news["category_code"].nunique()):
        index = np.where(news["category_code"].values == c)[0]
        print(str(c) + "-" + str(len(index)))
        index_val = random.sample(list(index), 16)
        index_for_validation.extend(index_val)

    data_validate = news.iloc[index_for_validation]
    data_train = news.drop(index_for_validation)
    data_train = data_train.reset_index(drop=True)

    index_for_train = []
    for c in range(data_train["category_code"].nunique()):
        index = np.where(data_train["category_code"].values == c)[0]
        index_train = random.sample(list(index), 950)
        index_for_train.extend(index_train)
    data_train = data_train.iloc[index_for_train]
    data_train = data_train.reset_index(drop=True)

    text_train = list(data_train["text"])
    labels_train = list(data_train["category_code"])

    text_validate = list(data_validate["text"])
    labels_validate = list(data_validate["category_code"])

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text_train)
    tokenizer.fit_on_texts(text_validate)
    save_tokenizer(tokenizer)
    sequences_train = tokenizer.texts_to_sequences(text_train)
    sequences_validate = tokenizer.texts_to_sequences(text_validate)

    data_train = pad_sequences(sequences_train, maxlen=maxlen)
    labels_train = np.asarray(labels_train)

    data_validate = pad_sequences(sequences_validate, maxlen=maxlen)
    labels_validate = np.asarray(labels_validate)

    # Randomly get training and validation samples
    indices = np.arange(data_train.shape[0])
    np.random.shuffle(indices)
    data_train = data_train[indices]
    labels_train = labels_train[indices]
    x_train = data_train
    y_train = labels_train

    x_val = data_validate
    y_val = labels_validate

    print("Size of train: " + str(len(x_train)))
    print("Size of validate: " + str(len(x_val)))

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    model = create_model()
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    filepath = 'bbc_news_classfication_model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), shuffle=True,
                        callbacks=[checkpoint])


def create_model():
    embedding_dim = 100
    # visible = Input(shape=(maxlen,))
    # hidden1 = Embedding(max_words, embedding_dim, input_length=maxlen)(visible)
    # flatten1 = Flatten()(hidden1)
    # hidden2 = Dense(256, kernel_initializer='uniform')(flatten1)
    # hidden2activation = LeakyReLU(alpha=.01)(hidden2)
    # hidden3 = Dense(128, kernel_initializer='uniform')(hidden2activation)
    # hidden3activation = LeakyReLU(alpha=.01)(hidden3)
    # hidden23 = Concatenate()([hidden2, hidden3activation])
    # output = Dense(20, kernel_initializer='uniform', activation='softmax')(hidden23)
    # model = Model(inputs=visible, outputs=output)

    visible = Input(shape=(maxlen,))
    hidden1 = Embedding(max_words, embedding_dim, input_length=maxlen)(visible)
    # hidden2 = LSTM(128,return_sequences = True)(hidden1)
    hidden2 = GRU(40)(hidden1)
    # hidden3 = LSTM(64)(hidden2)
    output = Dense(20, kernel_initializer='uniform', activation='softmax')(hidden2)
    model = Model(inputs=visible, outputs=output)

    # model = Sequential()
    # model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    # model.add(Flatten())
    # model.add(Dense(128, kernel_initializer='normal'))
    # model.add(LeakyReLU(alpha=.01))
    # model.add(Dense(64, kernel_initializer='normal'))
    # model.add(LeakyReLU(alpha=.01))
    # model.add(Dense(20, kernel_initializer='normal', activation='softmax'))

    model.summary()
    return model


def predict_category(string):
    model = create_model()
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Classification")
    model.load_weights(os.path.join(root_dir, "bbc_news_classfication_model.h5"))
    tokenizer = load_tokenizer()
    sequences = tokenizer.texts_to_sequences([string])
    data = pad_sequences(sequences, maxlen=maxlen)
    predictions = np.asarray(model.predict(data)).argmax()
    return categories[predictions]


def save_tokenizer(tokenizer):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer():
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Classification")
    with open(os.path.join(root_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def save_dictionary(dictionary):
    with open('dictionary.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dictionary():
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Classification")
    with open(os.path.join(root_dir, 'dictionary.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


main()
# main()
# predict("TEsting 1 2 3")

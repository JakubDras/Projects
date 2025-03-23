import keras_preprocessing.sequence
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import os
import numpy as np

with tf.device('/GPU:0'):
    VOCAB_SIZE = 88584
    MAXLEN = 250

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

    # print(train_data[2])
    # print(train_labels[2])
    train_data = keras_preprocessing.sequence.pad_sequences(train_data, MAXLEN)
    test_data = keras_preprocessing.sequence.pad_sequences(test_data, MAXLEN)

    # MODEL

    input_layer = Input(shape=MAXLEN)

    embedding_layer = layers.Embedding(VOCAB_SIZE, 32)(input_layer)

    lstm_layer = layers.LSTM(32)(embedding_layer)

    output_layer = layers.Dense(1, activation="sigmoid")(lstm_layer)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    print(model.summary())

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

    history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

    results = model.evaluate(test_data, test_labels)
    print(results)

    word_index = imdb.get_word_index()


    def encode_text(text):
        tokens = keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [word_index[word] if word in word_index else 0 for word in tokens]
        return keras_preprocessing.sequence.pad_sequences([tokens], MAXLEN)[0]


    text = "that movie was just amazing, so amazing"
    encoded = encode_text(text)
    print(encoded)

    reverse_word_index = {value: key for (key, value) in word_index.items()}


    def decode_integers(integers):
        PAD = 0
        text = ""
        for num in integers:
            if num != PAD:
                text += reverse_word_index[num] + " "

        return text[:-1]


    print(decode_integers(encoded))


    def predict(text):
        encoded_text = encode_text(text)
        pred = np.zeros((1, 250))
        pred[0] = encoded_text
        result = model.predict(pred)
        print(result[0])


    positive_review = "At first i didn't like this movie, but then at the and I loved it"
    predict(positive_review)

    negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
    predict(negative_review)
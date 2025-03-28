import keras_preprocessing.sequence
from keras.preprocessing import sequence
import keras
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


with tf.device('/GPU:0'):
    df = pd.read_csv("IMDB Dataset.csv")

    # print(df.info())

    # print(df['review'][0])

    dlugosc_zdan = []
    corpus = []

    for i in range(0,49999):
        tokens = keras.preprocessing.text.text_to_word_sequence(df["review"][i])
        dlugosc_zdan.append(len(tokens))
        corpus.append(df["review"][i])


    # plt.hist(dlugosc_zdan, bins=100) # 'bins' określa liczbę przedziałów
    # plt.xlabel("Wartości")
    # plt.ylabel("Liczba wystąpień")
    # plt.title("Histogram rozkładu dlugosci zdan")
    # plt.show()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    print(total_words)

    le = LabelEncoder()
    df['sentiment'] = le.fit_transform(df['sentiment'])

    rev = df['review']
    rev = rev.to_numpy()

    word_index = tokenizer.word_index
    # print(word_index)
    for i in range(0,len(rev)):
        tokens = keras.preprocessing.text.text_to_word_sequence(rev[i])
        token_list = sum(tokenizer.texts_to_sequences(tokens), [])
        rev[i] = token_list

    rev = keras_preprocessing.sequence.pad_sequences(rev, 250)

    print(rev[1])

    rev = pd.DataFrame(rev)

    print(rev.head())

    X = rev
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)


    # =================

    input_layer = Input(shape=250)

    embedding_layer = layers.Embedding(total_words, 32)(input_layer)

    lstm_layer = layers.LSTM(64)(embedding_layer)

    output_layer = layers.Dense(1, activation="sigmoid")(lstm_layer)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    print(model.summary())

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

    history = model.fit(X_train, y_train, epochs=10,batch_size= 20 ,validation_split=0.3)

    results = model.evaluate(X_test, y_test)
    print(results)



    def encode_text(text):
        tokens = keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [word_index[word] if word in word_index else 0 for word in tokens]
        return keras_preprocessing.sequence.pad_sequences([tokens], 250)[0]

    def predict(text):
        encoded_text = encode_text(text)
        pred = np.zeros((1, 250))
        pred[0] = encoded_text
        result = model.predict(pred)
        print(result[0])


    review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
    predict(review)


    review1 = ("A Waste of Time: 'Cosmic Catastrophe' promised intergalactic adventure, but delivered a tedious mess. The plot was nonsensical, the acting wooden, and the special effects laughable."
              " Even the soundtrack failed to impress, a jarring collection of generic synth tunes. Avoid this cinematic black hole; your time is better spent elsewhere.")
    predict(review1)

    review2 = ("The underwhelming spectacle that is 'Cosmic Catastrophe' fails to deliver on its premise. The plot is predictable, the acting wooden, and the special effects look like they were done on a potato."
               " Even the soundtrack couldn't save this cinematic train wreck. A complete waste of time and money. Avoid at all costs!")
    predict(review2)


    review3 = ("A cinematic triumph! 'Starlight Symphony' is a visual masterpiece, boasting breathtaking visuals and a captivating storyline. The acting is superb, the score unforgettable, and the emotional "
               "depth truly resonated. A must-see for any film lover. Prepare to be swept away by its magic!")
    predict(review3)

    review4 = ("A captivating cinematic experience! 'Echoes of the Past' masterfully blends mystery and romance. The acting is superb, the cinematography breathtaking, and the soundtrack hauntingly beautiful. "
               "A nuanced plot keeps you guessing until the very end, leaving a lasting impression. Highly recommended!")
    predict(review4)

    review5 = ("'Crimson Peak' boasts stunning visuals and a compelling central performance. However, the plot feels convoluted at times, and some supporting characters lack depth. While "
               "visually impressive, the narrative inconsistencies prevent it from being truly great. A mixed bag, ultimately.")
    predict(review5)

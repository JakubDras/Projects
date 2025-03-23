import pandas as pd
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers



def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year,month=month,day=day)

with tf.device('/GPU:0'):
    df = pd.read_csv("MSFT.csv")
    df = df[['Date', 'Close']]
    # print(df)

    df['Date'] = df['Date'].apply(str_to_datetime)
    # df.index = df.pop('Date')
    # plt.plot(df['Date'], df['Close'])
    # plt.show()

    preped_df = pd.DataFrame(columns=['Date','t-7','t-6','t-5','t-4','t-3','t-2','t-1','t'])
    for i in range(7,len(df)):
        preped_df.loc[i-7] = [df.loc[i,'Date'],df.loc[i-7,'Close'],df.loc[i-6,'Close'],df.loc[i-5,'Close'],df.loc[i-4,'Close'],df.loc[i-3,'Close'],df.loc[i-2,'Close'],df.loc[i-1,'Close'],df.loc[i,'Close']]

    # print(preped_df)
    # print(preped_df.info())

    preped_df = preped_df.iloc[int(len(preped_df) * 0.96):]
    df_dates = preped_df.pop('Date')
    df_target = preped_df.pop('t')

    print(preped_df)
    print(df_dates)
    print(df_target)

    df_dates.to_numpy()
    df_target.to_numpy()
    preped_df.to_numpy()

    # print(df_dates)
    # print(df_target)
    # print(preped_df)

    # SKONCZYLEM TU SAMEMU PISAĆ
    q_80 = int(len(df_dates) * .80)
    q_90 = int(len(df_dates) * .90)

    dates_train, X_train, y_train = df_dates[:q_80], preped_df[:q_80], df_target[:q_80]

    dates_val, X_val, y_val = df_dates[q_80:q_90], preped_df[q_80:q_90], df_target[q_80:q_90]
    dates_test, X_test, y_test = df_dates[q_90:], preped_df[q_90:], df_target[q_90:]

    # plt.plot(dates_train, y_train)
    # plt.plot(dates_val, y_val)
    # plt.plot(dates_test, y_test)
    #
    # plt.legend(['Train', 'Validation', 'Test'])
    # plt.show()


    model = Sequential([layers.Input((7, 1)),
                        layers.LSTM(64,return_sequences=True),
                        layers.LSTM(64),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50,batch_size=16)




    train_predictions = model.predict(X_train).flatten()

    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.legend(['Training Predictions', 'Training Observations'])
    plt.show()

    val_predictions = model.predict(X_val).flatten()

    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.legend(['Validation Predictions', 'Validation Observations'])
    plt.show()

    test_predictions = model.predict(X_test).flatten()

    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Testing Predictions', 'Testing Observations'])
    plt.show()

    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Training Predictions',
                'Training Observations',
                'Validation Predictions',
                'Validation Observations',
                'Testing Predictions',
                'Testing Observations'])
    plt.show()
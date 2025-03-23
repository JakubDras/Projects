import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import seaborn as sns
from sklearn.metrics import confusion_matrix

def F1(prec,recall):
    return 2*(prec * recall)/(prec + recall)

with tf.device('/GPU:0'):
    df = pd.read_csv('ArgTrenigEmot.csv')
    test = pd.read_csv('EmotionDettest.csv')

    test = test[test['emotion'] != 'disgusted']
    test = test[test['emotion'] != 'fearful']
    happyT = test[test['emotion'] == 'happy']
    test = test[test['emotion'] != 'happy']
    happyT = happyT.drop(happyT.index[::2]).reset_index(drop=True)
    test = pd.concat([test, happyT], ignore_index=True)

    sad_rowst = test[test['emotion'] == 'sad']
    num_to_remove = int(len(sad_rowst) * 0.25)
    rows_to_remove = sad_rowst.sample(n=num_to_remove, random_state=1)
    test = test.drop(rows_to_remove.index)

    neutral_rowst = test[test['emotion'] == 'neutral']
    num_to_remove = int(len(neutral_rowst) * 0.25)
    rows_to_remove = neutral_rowst.sample(n=num_to_remove, random_state=1)
    test = test.drop(rows_to_remove.index)


    df = df[df['emotion'] != 'disgusted']
    df = df[df['emotion'] != 'fearful']

    sad_rows = df[df['emotion'] == 'sad']
    num_to_remove = int(len(sad_rows) * 0.25)
    rows_to_remove = sad_rows.sample(n=num_to_remove, random_state=1)
    df = df.drop(rows_to_remove.index)

    happy_rows = df[df['emotion'] == 'happy']
    num_to_remove = int(len(happy_rows) * 0.6)
    rows_to_remove = happy_rows.sample(n=num_to_remove, random_state=1)
    df = df.drop(rows_to_remove.index)

    neutral_rows = df[df['emotion'] == 'neutral']
    num_to_remove = int(len(neutral_rows) * 0.6)
    rows_to_remove = neutral_rows.sample(n=num_to_remove, random_state=1)
    df = df.drop(rows_to_remove.index)

    df = df.sample(frac=1).reset_index(drop=True)
    x_train = df.drop('emotion', axis=1)
    y_train = df['emotion']

    test = test.sample(frac=1).reset_index(drop=True)
    x_test = test.drop('emotion', axis=1)
    y_test = test['emotion']


    y_train = pd.get_dummies(y_train, columns=["emotion"], dtype=int)
    y_test = pd.get_dummies(y_test, columns=["emotion"], dtype=int)

    print(test["emotion"].value_counts())
    print(df['emotion'].unique())
    print(df['emotion'].value_counts())
    x_train = x_train.to_numpy().reshape(78303, 48,48)
    x_test = x_test.to_numpy().reshape(4537, 48, 48)

    # ===========================================
    # MODEL

    inputs = Input(shape=(48, 48, 1))

    x = layers.Conv2D(250, (4, 4), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.40)(x)  # Zmiana Dropout na SpatialDropout2D

    x = layers.Conv2D(250, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.40)(x)

    sc1 = layers.Add()([inputs, x])

    x = layers.Conv2D(250, (3, 3), padding='same')(sc1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.40)(x)

    x = layers.Conv2D(250, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.40)(x)

    mx = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(250, (3, 3), padding='same')(mx)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.40)(x)

    x = layers.Conv2D(250, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.40)(x)

    sc1 = layers.Add()([mx, x])

    x = layers.Conv2D(250, (3, 3), padding='same')(sc1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.4)(x)

    mx2 = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(250, (3, 3), padding='same')(mx2)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.3)(x)

    sc1 = layers.Add()([mx2, x])

    x = layers.Conv2D(250, (2, 2), padding='same')(sc1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.3)(x)

    x13 = layers.Flatten()(x)
    x13 = layers.Dropout(0.5)(x13)

    x13 = layers.Dense(164)(x13)
    x13 = layers.BatchNormalization()(x13)
    x13 = layers.Activation('relu')(x13)
    x13 = layers.Dropout(0.5)(x13)

    x14 = layers.Dense(82)(x13)
    x14 = layers.BatchNormalization()(x14)
    x14 = layers.Activation('relu')(x14)
    x14 = layers.Dropout(0.3)(x14)

    outputs = layers.Dense(5, activation='softmax')(x14)

    model = models.Model(inputs=inputs, outputs=outputs)

    adamod = tf.keras.optimizers.Adam(
        learning_rate=0.0001
    )
    model.compile(optimizer=adamod,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', 'Precision', 'Recall'])

    history = model.fit(x_train, y_train, epochs=40, batch_size=8,
                        validation_data=(x_test, y_test))

    test_lossKuba, test_accKuba , test_precisionKuba, test_recallKuba = model.evaluate(x_test, y_test, verbose=4)
    print("Model Kuba:")
    print("acc: ", test_accKuba)
    print("loss: ", test_lossKuba)
    print("precision: ", test_precisionKuba)
    print("recall: ", test_recallKuba)
    print("F1: ", F1(test_precisionKuba,test_recallKuba))

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_true_classes = np.argmax(y_test, axis=-1)

    unique_classes = np.unique(y_true_classes)
    print(unique_classes)

    class_names = [ 'angry', 'happy','neutral', 'sad', 'surprised']
    correct_class_names = [class_names[i] for i in unique_classes]
    print(correct_class_names)

    cm = confusion_matrix(y_true_classes, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=correct_class_names, yticklabels=correct_class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

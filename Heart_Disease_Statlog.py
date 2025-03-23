import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/content/Heart_disease_statlog.csv")

print(df.info())
print(df.head(10))

def norm(df,Column_name):
  return (df[Column_name] - df[Column_name].min())/df[Column_name].max()

df['age'] = norm(df,'age')
df['cp'] = norm(df,'cp')
df['trestbps'] = norm(df,'trestbps')
df['chol'] = norm(df,'chol')
df['thalach'] = norm(df,'thalach')
df['oldpeak'] = norm(df,'oldpeak')
df['restecg'] = norm(df,'restecg')
df['ca'] = norm(df,'ca')
df['thal'] = norm(df,'thal')

print(df.head())

y_df = df.pop('target')
label_encoder = LabelEncoder()
y_df = label_encoder.fit_transform(y_df)
X_train, X_eval, y_train, y_eval = train_test_split(df, y_df, test_size=0.2, shuffle=True, random_state=42)

plt.figure(figsize=(7, 3))
sns.countplot(x=y_df, color="blue")
plt.title("Distribution of target")
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(13,)),
    keras.layers.Dense(45, activation="elu"),
    keras.layers.Dense(45, activation="elu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="Nadam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_eval, y_eval)
print('Test accuracy:', test_acc)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_eval)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = y_eval

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
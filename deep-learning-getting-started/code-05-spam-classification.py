# Setup

import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


# downloaded /root/nltk_data/


from nltk.corpus import stopwords

nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


# 5.2 Creating Text Representations


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# Load Spam Data and review content
spam_data = pd.read_csv("Spam-Classification.csv")

print("\nLoaded Data :\n------------------------------------")
print(spam_data.head())

# Separate feature and target data
spam_classes_raw = spam_data["CLASS"]
spam_messages = spam_data["SMS"]


import nltk
import tensorflow as tf


# Custom tokenizer to remove stopwords and use lemmatization
def customtokenize(str):
    # Split string as tokens
    tokens = nltk.word_tokenize(str)
    # Filter for stopwords
    nostop = list(filter(lambda token: token not in stopwords.words("english"), tokens))
    # Perform lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in nostop]
    return lemmatized


from sklearn.feature_extraction.text import TfidfVectorizer

# Build a TF-IDF Vectorizer model
vectorizer = TfidfVectorizer(tokenizer=customtokenize)

# Transform feature input to TF-IDF
tfidf = vectorizer.fit_transform(spam_messages)
# Convert TF-IDF to numpy array
tfidf_array = tfidf.toarray()

# Build a label encoder for target variable to convert strings to numeric values.
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
spam_classes = label_encoder.fit_transform(spam_classes_raw)

# Convert target to one-hot encoding vector
spam_classes = tf.keras.utils.to_categorical(spam_classes, 2)

print("TF-IDF Matrix Shape : ", tfidf.shape)
print("One-hot Encoding Shape : ", spam_classes.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    tfidf_array, spam_classes, test_size=0.10
)


# 5.3 Building and Evaluating the Model


from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

# Setup Hyper Parameters for building the model
NB_CLASSES = 2
N_HIDDEN = 32

model = tf.keras.models.Sequential()

model.add(
    keras.layers.Dense(
        N_HIDDEN,
        input_shape=(X_train.shape[1],),
        name="Hidden-Layer-1",
        activation="relu",
    )
)

model.add(keras.layers.Dense(N_HIDDEN, name="Hidden-Layer-2", activation="relu"))

model.add(keras.layers.Dense(NB_CLASSES, name="Output-Layer", activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()


# Make it verbose so we can see the progress
VERBOSE = 1

# Setup Hyper Parameters for training
BATCH_SIZE = 256
EPOCHS = 10
VALIDATION_SPLIT = 0.2

print("\nTraining Progress:\n------------------------------------")

history = model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
)

print("\nAccuracy during Training :\n------------------------------------")
import matplotlib.pyplot as plt

pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy improvements with Epoch")
plt.show()

print("\nEvaluation against Test Dataset :\n------------------------------------")
model.evaluate(X_test, Y_test)


# 5.4 Predicting for Text


# Predict for multiple samples using batch processing

# Convert input into IF-IDF vector using the same vectorizer model
predict_tfidf = vectorizer.transform(
    ["FREE entry to a fun contest", "Yup I will come over"]
).toarray()

print(predict_tfidf.shape)

# Predict using model
prediction = np.argmax(model.predict(predict_tfidf), axis=1)
print("Prediction Output:", prediction)

# Print prediction classes
print("Prediction Classes are ", label_encoder.inverse_transform(prediction))

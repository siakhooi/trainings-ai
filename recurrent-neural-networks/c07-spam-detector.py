## 07.01. Loading Spam Data

import pandas as pd
import os


#Load Spam Data and review content
spam_data = pd.read_csv("Spam-Classification.csv")

print("\nLoaded Data :\n------------------------------------")
print(spam_data.head())

#Separate feature and target data
spam_classes_raw = spam_data["CLASS"]
spam_messages = spam_data["SMS"]



## 07.02. Preprocessing Spam Data


#Build a label encoder for target variable to convert strings to numeric values.
from sklearn import preprocessing
import tensorflow as tf

label_encoder = preprocessing.LabelEncoder()
spam_classes = label_encoder.fit_transform(
                                spam_classes_raw)

#Convert target to one-hot encoding vector
spam_classes = tf.keras.utils.to_categorical(spam_classes,2)

print("One-hot Encoding Shape : ", spam_classes.shape)




#Preprocess data for spam messages
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Max words in the vocabulary for this dataset
VOCAB_WORDS=10000
#Max sequence length for word sequences
MAX_SEQUENCE_LENGTH=100

#Create a vocabulary with unique words and IDs
spam_tokenizer = Tokenizer(num_words=VOCAB_WORDS)
spam_tokenizer.fit_on_texts(spam_messages)


print("Total unique tokens found: ", len(spam_tokenizer.word_index))
print("Example token ID for word \"me\" :", spam_tokenizer.word_index.get("me"))

#Convert sentences to token-ID sequences
spam_sequences = spam_tokenizer.texts_to_sequences(spam_messages)

#Pad all sequences to fixed length
spam_padded = pad_sequences(spam_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print("\nTotal sequences found : ", len(spam_padded))
print("Example Sequence for sentence : ", spam_messages[0] )
print(spam_padded[0])






#Split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
                                    spam_padded,spam_classes,test_size=0.2)





## 07.03. Building the embeddding matrix



#Load the pre-trained embeddings

import numpy as np

#Read pretrained embeddings into a dictionary
glove_dict = {}

#Loading a 50 feature (dimension) embedding with 6 billion words
with open('glove.6B.50d.txt', "r", encoding="utf8") as glove_file:
    for line in glove_file:

        emb_line = line.split()
        emb_token = emb_line[0]
        emb_vector = np.array(emb_line[1:], dtype=np.float32)

        if emb_vector.shape[0] == 50:
            glove_dict[emb_token] = emb_vector

print("Dictionary Size: ", len(glove_dict))
print("\n Sample Dictionary Entry for word \"the\" :\n", glove_dict.get("the"))




#We now associate each token ID in our data set vocabulary to the corresponding embedding in Glove
#If the word is not available, then embedding will be all zeros.

#Matrix with 1 row for each word in the data set vocubulary and 50 features

vocab_len = len(spam_tokenizer.word_index) + 1

embedding_matrix = np.zeros((vocab_len, 50))

for word, id in spam_tokenizer.word_index.items():
    try:
        embedding_vector = glove_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[id] = embedding_vector
    except:
        pass

print("Size of Embedding matrix :", embedding_matrix.shape)
print("Embedding Vector for word \"me\" : \n", embedding_matrix[spam_tokenizer.word_index.get("me")])






## 07.04. Build the Spam Model with Embeddings



#Create a model
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from keras.layers import LSTM,Dense

#Setup Hyper Parameters for building the model
NB_CLASSES=2

model = tf.keras.models.Sequential()

model.add(keras.layers.Embedding(vocab_len,
                                 50,
                                 name="Embedding-Layer",
                                 weights=[embedding_matrix],
                                 input_length=MAX_SEQUENCE_LENGTH,
                                 trainable=True))

#Add LSTM Layer
model.add(LSTM(256))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(NB_CLASSES,
                             name='Output-Layer',
                             activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()





#Make it verbose so we can see the progress
VERBOSE=1

#Setup Hyper Parameters for training
BATCH_SIZE=256
EPOCHS=10
VALIDATION_SPLIT=0.2

print("\nTraining Progress:\n------------------------------------")

history=model.fit(X_train,
          Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)

print("\nEvaluation against Test Dataset :\n------------------------------------")
model.evaluate(X_test,Y_test)







## 07.05. Predicting Spam




# Two input strings to predict
input_str=["Unsubscribe send GET EURO STOP to 83222",
            "Yup I will come over"]

#Convert to sequence using the same tokenizer as training
input_seq = spam_tokenizer.texts_to_sequences(input_str)
#Pad the input
input_padded = pad_sequences(input_seq, maxlen=MAX_SEQUENCE_LENGTH)

#Predict using model
prediction=np.argmax( model.predict(input_padded), axis=1 )
print("Prediction Output:" , prediction)

#Print prediction classes
print("Prediction Classes are ", label_encoder.inverse_transform(prediction))

# Poetry generation challenge

# This notebook serves as a challenge on how to create poetry like Shakespeare by leveraging RNNs(LSTMs). We'll be using the Shakerpeare poetry as the training data and then use the trained network to predict the next words.


##import the required libraries and APIs
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)


## Step 1: Create a corpus

##download data from this url
# !wget --no-check-certificate \
#     https://raw.githubusercontent.com/dswh/lil_nlp_with_tensorflow/main/sonnets.txt \
#     -O /tmp/sonnet.txt


##printing the text
shakespeare_text = open("sonnets.txt").read()
print(len(shakespeare_text))

##create corpus by lowering the letters and splitting the text by \n
corpus = shakespeare_text.lower().split("\n")
print(corpus)


## Set up the tokenizer


##set up tokenizer
tokenizer = Tokenizer()


tokenizer.fit_on_texts(corpus)

##calculate vocabulary size - be mindful of the <oov> token
vocab_size = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(vocab_size)


##create sequences of
input_sequences = []
for line in corpus:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        n_gram_sequence = tokens[: i + 1]
        input_sequences.append(n_gram_sequence)


##pad sequences
max_seq_len = max([len(i) for i in input_sequences])
input_seq_array = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")
)


##creating features(X) and label(y)
X = input_seq_array[:, :-1]
labels = input_seq_array[:, -1]

##one-hot encode the labels to get y - since it is actually just a classification problem
y = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)


## Define the LSTM model


model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, 120, input_length=max_seq_len - 1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(120)),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ]
)

##define the learning rate - step size for optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
history = model.fit(X, y, epochs=200, verbose=1)


## Visualise the metrics


import matplotlib.pyplot as plt


def plot_metric(history, metric):
    plt.plot(history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.show()


plot_metric(history, "accuracy")


## Generate new text


seed_text = "It was a cold night."
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
    # predicted = model.predict_classes(token_list, verbose=0)
    predict_x = model.predict(token_list)
    predicted = np.argmax(predict_x,axis=1)

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)

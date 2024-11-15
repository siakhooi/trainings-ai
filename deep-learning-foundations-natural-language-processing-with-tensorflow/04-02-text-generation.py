# Introduction to Text generation

# This notebook explains how we can split a given corpus of data into features and labels and then train a neural network to predict the next word in a sentence.

# 1. Create a corpus - break the text down to list of sentences.
# 2. Create a word_index(vocabulary) from the text.
# 3. Tokenize the data and create n-gram sequence for each sequence of the corpus.
# 4. Pad those sequences.
# 5. Segregate features from the sequences by reserving the last element of the array as labels.


##import the required libraries and APIs
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)

## Step 1: Create a corpus


data = "October arrived, spreading a damp chill over the grounds and into the castle.\n Madam Pomfrey, the nurse, was kept busy by a sudden spate of colds among the staff and students.\n Her Pepperup potion worked instantly, though it left the drinker smoking at the ears for several hours afterward. Ginny Weasley, who had been looking pale, was bullied into taking some by Percy.\n The steam pouring from under her vivid hair gave the impression that her whole head was on fire.\n Raindrops the size of bullets thundered on the castle windows for days on end; the lake rose, the flower beds turned into muddy streams, and Hagrid's pumpkins swelled to the size of garden sheds.\n Oliver Wood's enthusiasm for regular training sessions, however, was not dampened, which was why Harry was to be found, late one stormy Saturday afternoon a few days before Halloween, returning to Gryffindor Tower, drenched to the skin and splattered with mud."


##instantiate tokenizer
tokenizer = Tokenizer()

##create corpus by lowering the letters and splitting the text by \n
corpus = data.lower().split("\n")
print(corpus)

## Step 2: Train the tokenizer and create word encoding dictionary


tokenizer.fit_on_texts(corpus)

##calculate vocabulary size - +1 for <oov> token
vocab_size = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(vocab_size)


## Step 3: Create N-gram sequence


##create n-gram sequences of each text sequence
input_sequences = []
for line in corpus:
    tokens = tokenizer.texts_to_sequences([line])[0]  # get all the tokens of the sequence
    for i in range(1, len(tokens)):  # create n-gram sequences
        n_gram_sequence = tokens[:i+1]
        input_sequences.append(n_gram_sequence)



##pad sequences
max_seq_len = max([len(i) for i in input_sequences])
input_seq_array = np.array(pad_sequences(input_sequences,
                                         maxlen=max_seq_len,
                                         padding='pre')
                        )




## Step 4: Extract features and labels


##creating features(X) and label(y)
X = input_seq_array[:, :-1]
labels = input_seq_array[:, -1]

##one-hot encode the labels to get y
y = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

print(tokenizer.word_index['mud'])
print(X[0])
print(y[0])


## Define the LSTM model


model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, 64, input_length=max_seq_len-1),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=500, verbose=1)


## Visualize metrics

import matplotlib.pyplot as plt


def plot_metric(history, metric):
  plt.plot(history.history[metric])
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.show()


plot_metric(history, 'accuracy')

## Generate new text


seed_text = "It was a cold night."

##add number of words you want to predict
next_words = 100

##run the loop to predict and concatenate the word
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

    ##predict the class using the trained model
    #predicted = model.predict_classes(token_list, verbose=0)
    predict_x = model.predict(token_list)
    predicted = np.argmax(predict_x,axis=1)

    output_word = ""
    for word, index in tokenizer.word_index.items():
        ##reference the predicted class with the vocabulary
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)

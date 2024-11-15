from tensorflow.keras.preprocessing.text import Tokenizer

##define list of sentences to tokenize
train_sentences = [
             'It is a sunny day',
             'It is a cloudy day',
             'Will it rain today?'

]

# Train the tokenizer

##set up the tokenizer
tokenizer = Tokenizer(num_words=100)

##train the tokenizer on training sentences
tokenizer.fit_on_texts(train_sentences)

##store word index for the words in the sentence
word_index = tokenizer.word_index

# Create sequences

##create sequences using tokenizer
sequences = tokenizer.texts_to_sequences(train_sentences)

##print word index dictionary and sequences
print(f"Word index -->{word_index}")
print(f"Sequences of words -->{sequences}")


##print sample sentence and sequence
print("sample sentence and sequence")
print(train_sentences[0])
print(sequences[0])


# Tokenizing new data using the same tokenizer

new_sentences = [
                 'Will it be raining today?',
                 'It is a pleasant day.'
]

new_sequences = tokenizer.texts_to_sequences(new_sentences)

print("new sequence with missing value:")
print(new_sentences)  ## missing value
print(new_sequences)  ## missing value


# Replacing newly encountered words with special values

##set up the tokenizer again with oov_token
tokenizer = Tokenizer(num_words=100, oov_token = "<oov>")
##train the new tokenizer on training sentences
tokenizer.fit_on_texts(train_sentences)
##store word index for the words in the sentence
word_index = tokenizer.word_index


##create sequences of the new sentences
new_sequences = tokenizer.texts_to_sequences(new_sentences)

print("new sequence with oov:")
print(word_index)
print(new_sentences)
print(new_sequences)

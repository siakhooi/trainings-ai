##import the required APIs
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_sentences = [
    "It will rain",
    "The weather is cloudy!",
    "Will it be raining today?",
    "It is a super hot day!",
]

##set up the tokenizer again with oov_token
tokenizer = Tokenizer(num_words=100, oov_token="<oov>")
##train the tokenizer on training sentences
tokenizer.fit_on_texts(train_sentences)
##store word index for the words in the sentence
word_index = tokenizer.word_index


##create sequences
sequences = tokenizer.texts_to_sequences(train_sentences)

# Pad the sequence

##pad sequences
padded_seqs = pad_sequences(sequences)

print(word_index)
print(train_sentences)
print(sequences)
print(padded_seqs)

# Customising your padded sequence with parameters

##pad sequences with padding type, max length and truncating parameters
padded_seqs = pad_sequences(
    sequences,
    padding="post",
    maxlen=5,
    truncating="post",
)

print(padded_seqs)

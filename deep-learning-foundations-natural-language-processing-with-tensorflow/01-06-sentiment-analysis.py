##read the data using the pandas library
import pandas as pd

data = pd.read_json("./x1.json")
data.head()

# Segregating the headlines

##create lists to store the headlines and labels
headlines = list(data['headline'])
labels = list(data['is_sarcastic'])

##import the required APIs
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

##set up the tokenizer
tokenizer = Tokenizer(oov_token="<oov>")
tokenizer.fit_on_texts(headlines)

word_index = tokenizer.word_index
print(word_index)

##create sequences of the headlines
seqs = tokenizer.texts_to_sequences(headlines)

##post-pad sequences
padded_seqs = pad_sequences(seqs, padding="post")

##printing padded sequences sample
print(padded_seqs[0])

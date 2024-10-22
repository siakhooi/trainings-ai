from rank_bm25 import BM25Okapi
from sklearn.datasets import fetch_20newsgroups
import string

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data  # A list of documents (newsgroup posts)

# Preprocess the documents
def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

# Tokenize the documents
tokenized_docs = [preprocess(doc) for doc in documents]

# Create a BM25 object
bm25 = BM25Okapi(tokenized_docs)

# Example search query
query = "What are some of the good gun manufacturing brands?"
tokenized_query = preprocess(query)

# Perform search
doc_scores = bm25.get_scores(tokenized_query)

# Get top N documents
top_n = 3
top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]

# Display top N results
for idx in top_doc_indices:
    print(f"Document ID: {newsgroups.filenames[idx]}, Score: {doc_scores[idx]}\nDocument: {documents[idx][:600]}...\n")

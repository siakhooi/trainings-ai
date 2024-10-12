import re
import collections

def count_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        content=infile.read()
        words = re.findall(r"[0-9a-zA-Z'-]+", content)

    words=[word.upper() for word in words]

    wordcount=collections.Counter(words)

    print(f"\nTotal Words: {len(words)}")
    print("\nTop 20 Words:")
    for [k, v] in wordcount.most_common(20):
        print(f"{k:7} {v:5}")

def count_words1(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        content=infile.read()
        words = re.findall(r"[0-9a-zA-Z'-]+", content)

    wordcount=collections.defaultdict(int)
    for word in words:
        wordcount[word.upper()]+=1

    print(f"\nTotal Words: {len(words)}")
    print("\nTop 20 Words:")
    sorted_word_count=dict(sorted(wordcount.items(), key=lambda item: item[1], reverse=True))

    i=0
    for _, (k,v) in enumerate(sorted_word_count.items()):
        print(f"{k:7} {v:5}")
        i+=1
        if i==20:
            break

# commands used in solution video for reference
if __name__ == '__main__':
#    count_words('shakespeare.txt')
    count_words('level-up-python-3210418-main/src/10 Count Unique Words/shakespeare.txt')

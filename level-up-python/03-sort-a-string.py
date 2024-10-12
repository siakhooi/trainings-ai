def sort_words(s):
    return " ".join(sorted(s.split(), key=str.casefold))

if __name__ == '__main__':
    print(sort_words('banana ORANGE apple'))  # apple banana ORANGE

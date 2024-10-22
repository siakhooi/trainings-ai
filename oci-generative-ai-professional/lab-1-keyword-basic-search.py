# Simulated database of Wikipedia-like entries
articles = [
    {'title': 'Python (programming language)', 'link': 'https://en.wikipedia.org/wiki/Python_(programming_language)'},
    {'title': 'History of Python', 'link': 'https://en.wikipedia.org/wiki/History_of_Python'},
    {'title': 'Monty Python', 'link': 'https://en.wikipedia.org/wiki/Monty_Python'},
    {'title': 'Anaconda (Python distribution)', 'link': 'https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)'},
    {'title': 'Python molurus', 'link': 'https://en.wikipedia.org/wiki/Python_molurus'},
    {'title': 'Association football', 'link': 'https://en.wikipedia.org/wiki/Association_football'},
    {'title': 'FIFA World Cup', 'link': 'https://en.wikipedia.org/wiki/FIFA_World_Cup'},
    {'title': 'History of artificial intelligence', 'link': 'https://en.wikipedia.org/wiki/History_of_artificial_intelligence'},
    {'title': 'Football in England', 'link': 'https://en.wikipedia.org/wiki/Football_in_England'},
    {'title': 'Applications of artificial intelligence', 'link': 'https://en.wikipedia.org/wiki/Applications_of_artificial_intelligence'}
]

# Function to perform keyword search on the simulated database
def keyword_search(articles, keyword):
    # Convert keyword to lowercase for case-insensitive matching
    keyword = keyword.lower()
    # Search for the keyword in the titles of the articles
    results = [article for article in articles if keyword in article['title'].lower()]
    return results

# Example usage
keyword = input("Enter a keyword to search: ")
search_results = keyword_search(articles, keyword)

# Display the search results
for result in search_results:
    print(result['title'], result['link'])

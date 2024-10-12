# import json

# def save_dict(dict, filepath):
#     with open(filepath, 'w') as outfile:
#          json.dump(dict, outfile)

# def load_dict(filepath):
#     with open(filepath, 'r') as infile:
#       return json.loads(infile.read())

import pickle
def save_dict(dict, filepath):
    with open(filepath, 'wb') as outfile:
         pickle.dump(dict, outfile)

def load_dict(filepath):
    with open(filepath, 'rb') as infile:
      return pickle.load(infile)

# commands used in solution video for reference
if __name__ == '__main__':
    test_dict = {1: 'a', 2: 'b', 3: 'c'}
    save_dict(test_dict, 'test_dict.pickle')
    recovered = load_dict('test_dict.pickle')
    print(recovered)  # {1: 'a', 2: 'b', 3: 'c'}

import random
import secrets
master='level-up-python-3210418-main/src/11 Generate a Password/diceware.wordlist.asc'

def generate_passphrase(n):
    with open(master, 'r') as file:
        lines = file.readlines()[2:7778]
        word_list=[line.split()[1] for line in lines]

    words = [secrets.choice(word_list) for _ in range(n)]
    return ' '.join(words)

def generate_passphrase1(n):
    masterlist=[]
    extract_flag=False
    with open(master, 'r') as file:
        for line in file:
            line=line.strip()
            if extract_flag:
                if line=="":
                    break
                else:
                    s=line.split()
                    masterlist.append(s[1])
            elif line=="":
                extract_flag=True

    passwords=[]
    for _ in range(n):
        d=random.randint(0,len(masterlist))
        passwords.append(masterlist[d])

    return " ".join(passwords)


# commands used in solution video for reference
if __name__ == '__main__':
    print(generate_passphrase(7))
    print(generate_passphrase(7))

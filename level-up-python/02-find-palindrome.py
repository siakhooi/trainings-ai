def is_palindrome(s):
    s=[ c for c in s.lower() if c>='a' and c<='z']
    l=len(s)
    mid=l//2
    for i in range(mid):
        if s[i]!=s[l-i-1]:
            return False
    return True

def is_palindrome2(s):
    s=[ c for c in s.lower() if c>='a' and c<='z']
    f=s[::-1]
    return f==s

if __name__ == '__main__':
    print(is_palindrome2('hello world'))  # false
    print(is_palindrome2("Go hang a salami, I'm a lasagna hog."))  # true

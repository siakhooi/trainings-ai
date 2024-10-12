def get_prime_factors(n):
    if n <=1:
        return []
    t=2
    l=int(n**0.5)
    result=[]

    while n>1 and t<=l:
        if n%t==0:
            result+=[t]
            n/=t
        else:
            t+=1
    if not result:
        result +=[n]

    return result

if __name__ == '__main__':
    # 2,3,3,5,7
    print(get_prime_factors(630))
    # 13
    print(get_prime_factors(13))

    # print(get_prime_factors(1024))
    # print(get_prime_factors(72))
    # print(get_prime_factors(49))
    # print(get_prime_factors(27))
    # print(get_prime_factors(2))
    # print(get_prime_factors(1))
    # print(get_prime_factors(0))

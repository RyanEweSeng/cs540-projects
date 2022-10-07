import sys
import math

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the 26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    X=[]
    for i in range(26):
        X.insert(i, 0)
    
    with open (filename,encoding='utf-8') as f:
        for line in f:
            for char in line.upper().strip():
                if char.isalpha():
                    X[ord(char) - 65] = X[ord(char) - 65] + 1

    return X

def compute(X, p, i):
    return X[i] * math.log(p[i])

def big_func(language, X, p):
    prob = 0
    if language == 'e':
        prob = 0.6
    elif language == 's':
        prob = 0.4

    sum = 0
    for i in range(26):
        sum += compute(X, p, i)
    
    return math.log(prob) + sum

def final_func(bigF_e, bigF_s):
    delta = bigF_s - bigF_e

    if delta >= 100:
        return 0
    elif delta <= -100:
        return 1
    else:
        return 1 / (1 + math.exp(delta))

def main():
    vectors = get_parameter_vectors()

    X = shred('letter.txt')

    val_e = compute(X, vectors[0], 0)
    val_s = compute(X, vectors[1], 0)

    bigF_e = big_func('e', X, vectors[0])
    bigF_s = big_func('s', X, vectors[1])

    prob_english_given_txt = final_func(bigF_e, bigF_s)

    print('Q1')
    for i in range(26):
        print(chr(i + 65),X[i])

    print('Q2')
    print(f'{val_e:.4f}')
    print(f'{val_s:.4f}')

    print('Q3')
    print(f'{bigF_e:.4f}')
    print(f'{bigF_s:.4f}')

    print('Q4')
    print(f'{prob_english_given_txt:.4f}')

main()
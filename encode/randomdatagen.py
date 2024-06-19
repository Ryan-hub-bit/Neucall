import random


if __name__ == "__main__":
    text = 'abcdefghijklmnopqrstuvwxyz'
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    notation = vocab_size+1
    notasetlen = 3
    notaset = set()
    k = notation
    for _ in range(notasetlen):
        k+=1
        notaset.add(k)


    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    finalseq = []
    keys = []
    words = []
    totaln = random.randint(1,100)
    for _ in range(totaln):
        while True:
            newkey = []
            i = random.randint(1, 7)
            for _ in range(i):
                newkey.append(random.sample(notaset,1)[0])
            if newkey not in keys:
                keys.append(newkey)
                keys.append(newkey)
                break
    random.shuffle(keys)
    totaln *=2
    for t in range(totaln):
        s=[]
        i=random.randint(1,7)
        for _ in range(i):
            c = random.randint(1, vocab_size)
            s.append(c)
        words.append(s)
        finalseq+=s+[notation]+keys[t]+[notation]
    answers = dict()
    for t in range(totaln):
        if t not in answers:
            for k in range(t+1,totaln):
                if keys[t] == keys[k]:
                    answers[t]=k
                    answers[k]=t
                    break



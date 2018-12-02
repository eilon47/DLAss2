import numpy as np
from tagger1_utils import WORD_TO_VEC as W2V


def dis(w1,w2):
    d = np.max([float(np.linalg.norm(w1, 2) * np.linalg.norm(w2, 2)), 1e-8])
    n = np.dot(w1, w2)
    return n / d


def most_similar(word, k=5):
    similarities = []
    for w in W2V.keys():
        if w is word:
            continue
        similarities.append((w, dis(W2V[word],W2V[w])))
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)

    return top_k[1:k+1]


if __name__ == '__main__':
    for word in ["dog", "england", "john", "explode", "office"]:
        print "5 most simiar words to " + str(word) + ": "
        print "\n".join([str(t) for t in most_similar(word)])
        print "##################################"



import numpy as np
import tagger1_utils as utils
from tagger1_utils import WORD_TO_VEC as W2V

word_vec = "wordVectors.txt"
vocab = "vocab.txt"
vecs = np.loadtxt(word_vec)
fd = open(vocab,'r')
data = fd.readlines()
fd.close()


def dis(w1,w2):
    d = np.max([float(la.norm(w1, 2) * la.norm(w2, 2)), 1e-8])
    n = np.dot(u, v)
    return n / d

    vec1 = W2V[w1]
    vec2 = W2V[w2]
    p1 = np.dot(vec1, vec2)
    p2 = np.sqrt(np.dot(vec1, vec1))
    p3 = np.sqrt(np.dot(vec2, vec2))
    distance = np.divide(p1,np.multiply(p2,p3))
    return distance


def most_similar(word, k=5):
    similarities = []
    most_similar = []
    for w in W2V.keys():
        if w is word:
            continue
        distance = dis(word, w)
        similarities.append((w, distance))

    for i in range(k):

        min_dis = min(similarities.items(), key=lambda x: x[1])
        most_similar.append(min_dis)

    return similarities


if __name__ == '__main__':
    most_similar('dog')


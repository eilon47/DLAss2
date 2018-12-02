import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}

# Globals
wordVecotrFile = "wordVectors.txt"
vocabFile = "vocab.txt"
START_STR, END_STR, UNK = "SSSTARTTT", "EEENDDD", "UUUNKKK"
START_W_TAG, END_W_TAG = (START_STR, START_STR), (END_STR, END_STR)
E = np.loadtxt(wordVecotrFile)
WORD_TO_VEC = {word.strip():vector for word, vector in zip(open(vocabFile), E)}
TAGS, WORDS = set(), set()
T2I, I2T, W2I, I2W = dict(), dict(), dict(), dict()
is_pre_trained = False


# for part4
SUFF_LENGTH = 3
PREF_LENGTH = 3


def create_P2I(prefixes):
    return {pref: i for i, pref in enumerate(prefixes)}


def create_S2I(suffixes):
    return {suff: i for i, suff in enumerate(suffixes)}



def set_pre_trained(value):
    global is_pre_trained
    is_pre_trained = value


def read_data(fname, tagged_data=True, is_train=True, seperator=" "):
    """
    This function reads the data into list of sentences.
    if the data is tagged data and each word is in new line,
    each sentence will be list of words with their tags, as tuples.
    :param fname: file path
    :param tagged_data: if the data is tagged
    :param is_train: if the data is read for train
    :return: data , list of sentences
    """
    data = []
    sentence = []
    # For not tagged data
    print "Reading data from:", fname, " tagged data?", tagged_data, " is train?", is_train
    if not tagged_data:
        for line in file(fname):
            if line.strip() == "":
                data.append(" ".join(sentence))
                sentence = []
                continue
            if is_pre_trained:
                line = line.lower()
            sentence.append(line.strip())
        return data
    # For tagged data
    global TAGS, WORDS
    for line in file(fname):
        try:
            if is_pre_trained:
                line = line.lower()
            word, label = line.strip().split(seperator,1)
            sentence.append((word, label))
            if is_train:
                TAGS.add(label)
                WORDS.add(word)
        except ValueError:
            data.append(sentence)
            sentence = []
    if is_train:
        WORDS.add(UNK)
        TAGS.add(UNK)
    if len(sentence) is not 0 and sentence not in data:
        data.append(sentence)
    print "Finished reading data."
    return data


def initialize_indexes():
    """
    initialize the indexes and the words in the dictionaries
    :return:
    """
    global T2I, I2T, W2I, I2W, WORDS, TAGS
    WORDS.update([START_STR, END_STR])
    for i, word in enumerate(WORDS):
        W2I[word] = i
        I2W[i] = word
    for i, tag in enumerate(TAGS):
        T2I[tag] = i
        I2T[i] = tag


def create_windows(sentences, windows_length=2):
    """
    create windows for each sentence in the sentences
    :param sentences:
    :param windows_length: number of words from each side of the mid word
    :return:
    """
    print "Creating windows from", len(sentences), "sentences"
    windows = []
    mid_tags = []
    js = range(-windows_length, windows_length+1)
    for sentence in sentences:
        padded = pad_sentence(sentence, windows_length)
        for i in range(windows_length, len(padded)-(windows_length)):
            window = []
            word, tag = padded[i]
            for j in js:
                offset = i + j
                window.append(padded[offset])
            windows.append(index_window(window))
            mid_tags.append(tags_index(tag))
    print "created", len(windows), "windows and", len(mid_tags), "tags"
    return windows, mid_tags


def create_windows_without_tags(sentences, windows_length=2):
    """
       create windows for each sentence in the sentences
       :param sentences:
       :param windows_length: number of words from each side of the mid word
       :return:
       """
    print "creating windows for a file without tags"
    windows = []
    js = range(-windows_length, windows_length + 1)
    for sentence in sentences:
        padded = pad_sentence(sentence, windows_length)
        for i in range(windows_length, len(padded) - (windows_length)):
            window = []
            for j in js:
                offset = i + j
                window.append(padded[offset])
            windows.append(index_window(window, is_tagged=False))
    print "created ", len(windows), " windows "
    return windows


def index_window(window, is_tagged=True):
    """
    create list of indexes of each word in the window
    :param window:
    :param is_tagged:
    :return:
    """
    indexes = []
    if is_tagged:
        for w,t in window:
            indexes.append(words_index(w))
    else:
        for w in window:
            indexes.append(words_index(w))
    return indexes


def words_index(word):
    """
    returns the index of the word
    :param word:
    :return:
    """
    try:
        index = W2I[word]
        return index
    except KeyError:
        return W2I[UNK]


def tags_index(tag):
    """
    returns the tag's index
    :param tag:
    :return:
    """
    index = T2I[tag]
    return index


def from_index_to_tag(index):
    """
    returns the tag according to index or list of indexes
    :param index:
    :return:
    """
    if isinstance(index, int):
        return I2T[index]
    if isinstance(index, list):
        return [I2T[i] for i in index]


def pad_sentence(sentence, window_length, is_tagged=True):
    """
    pad sentences with start and end
    :param sentence:
    :param window_length:
    :param is_tagged:
    :return:
    """
    S_PAD = START_W_TAG if is_tagged else START_STR
    E_PAD = END_W_TAG if is_tagged else END_STR
    padded = []
    for i in range(window_length):
        padded.append(S_PAD)
    padded += sentence
    for i in range(window_length):
        padded.append(E_PAD)
    return padded


def plot_graph(plot_values, color, label):
    """
    create plot
    :param plot_values: x:y dictionary
    :param color: plot color
    :param label: name of the plot
    :return:
    """
    line1, = plt.plot(plot_values.keys(), plot_values.values(), color,
                      label=label)
    # drawing name of the graphs
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()


def tagged_window_to_words_and_tags(window):
    """
    returns the words and tags of a window
    :param window:
    :return:
    """
    words, tags = [], []
    for tup in window:
        w,t = tup
        words.append(w)
        tags.append(t)
    return words, tags





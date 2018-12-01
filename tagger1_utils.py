import numpy as np

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}

wordVecotrFile = "wordVectors.txt"
vocabFile = "vocab.txt"
START_STR, END_STR, UNK = "SSSTARTTT", "EEENDDD", "UUUNKKK"
START_W_TAG, END_W_TAG = (START_STR, START_STR), (END_STR, END_STR)

WORD_TO_VEC = {word.strip():np.asanyarray(vector.strip().split(" ")) for word, vector in zip(open(vocabFile), open(wordVecotrFile))}
TAGS, WORDS = set(), set()
T2I, I2T, W2I, I2W = dict(), dict(), dict(), dict()


def read_data(fname, tagged_data=True, is_train=True):
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
            sentence.append(line.strip())
        return data
    # For tagged data
    global TAGS, WORDS
    for line in file(fname):
        try:
            word, label = line.strip().split(" ",1)
            sentence.append((word, label))
            if is_train:
                TAGS.add(label)
                WORDS.add(word)
        except ValueError:
            data.append(sentence)
            sentence = []
    if is_train:
        WORDS.update([UNK, START_STR, END_STR])
        TAGS.update([UNK, START_STR, END_STR])
    if len(sentence) is not 0 and sentence not in data:
        data.append(sentence)
    print "Finished reading data."
    return data


def initialize_indexes():
    global T2I, I2T, W2I, I2W
    for i, word in enumerate(WORDS):
        W2I[word] = i
        I2W[i] = word
    for i, tag in enumerate(TAGS):
        T2I[tag] = i
        I2T[i] = tag


def words_index(word):
    try:
        index = W2I[word]
        return index
    except KeyError:
        return W2I[UNK]


def tags_index(tag):
    index = T2I[tag]
    return index


def pad_sentence(sentence, window_length, is_tagged=True):
    S_PAD = START_W_TAG if is_tagged else START_STR
    E_PAD = END_W_TAG if is_tagged else END_STR
    padded = []
    for i in range(window_length):
        padded.append(S_PAD)
    padded += sentence
    for i in range(window_length):
        padded.append(E_PAD)
    return padded


def create_windows(sentences, windows_length=2, with_tags=True):
    print "Creating windows from", len(sentences), "sentences"
    if with_tags:
        initialize_indexes()
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


def tagged_window_to_words_and_tags(window):
    words, tags = [], []
    for tup in window:
        w,t = tup
        words.append(w)
        tags.append(t)
    return words, tags


def index_window(window, is_tagged=True):
    indexes = []
    if is_tagged:
        for w,t in window:
            indexes.append(words_index(w))
    else:
        for w in window:
            indexes.append(words_index(w))
    return indexes


if __name__ == '__main__':
    data = read_data("samples/train_sample.txt")
    initialize_indexes()
    win, tags = create_windows(data)
    for w in win:
        print  w
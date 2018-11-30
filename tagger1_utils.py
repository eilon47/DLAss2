STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}

wordVecotrFile = "samples/wordVectors_sample.txt"
vocabFile = "samples/vocab_sample.txt"
START_STR, END_STR, UNK = "SSSTARTTT", "EEENDDD", "UUUNKKK"
START_W_TAG, END_W_TAG = (START_STR, START_STR), (END_STR, END_STR)

WORD_TO_VEC = {word.strip():vector.strip().split(" ") for word, vector in zip(open(vocabFile), open(wordVecotrFile))}
TAGS, WORDS = set(), set()
T2I, I2T, W2I, I2W = dict(), dict(), dict(), dict()


def read_data(fname, tagged_data=True):
    """
    This function reads the data into list of sentences.
    if the data is tagged data and each word is in new line,
    each sentence will be list of words with their tags, as tuples.
    :param fname: file path
    :param tagged_data: if the data is tagged
    :return: data , list of sentences
    """
    data = []
    sentence = []
    # For not tagged data
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
            TAGS.add(label)
            WORDS.add(word)
        except ValueError:
            data.append(sentence)
            sentence = []
    WORDS.add(START_STR)
    WORDS.add(END_STR)
    if len(sentence) is not 0 and sentence not in data:
        data.append(sentence)
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


def pad_sentence(sentence, window_length):
    padded = []
    for i in range(window_length):
        padded.append(START_W_TAG)
    padded += sentence
    for i in range(window_length):
        padded.append(END_W_TAG)
    return padded


def create_windows(sentences, windows_length=2):
    windows = []
    js = range(-windows_length, windows_length+1)
    for sentence in sentences:
        padded = pad_sentence(sentence, windows_length)
        for i in range(windows_length, len(padded)-(windows_length)):
            window = []
            for j in js:
                offset = i + j
                window.append(padded[offset])
            windows.append(window)
    return windows


def tagged_window_to_words_and_tags(window):
    words, tags = [], []
    for tup in window:
        w,t = tup
        words.append(w)
        tags.append(t)
    return words, tags


def index_window(window):
    indexes = []
    for w,t in window:
        indexes.append(words_index(w))
    return indexes


if __name__ == '__main__':
    for i in range(5):
        print WORD_TO_VEC.items()[i]
    fname = "samples/train_sample.txt"
    sentences = read_data(fname)
    initialize_indexes()
    windows = create_windows(sentences, windows_length=2)
    for win in windows:
        words, tags = tagged_window_to_words_and_tags(win)
        print " ".join(words), " ### " , " ".join(tags)
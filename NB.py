###
### DISCLAIMER: THIS IS A FULL CODE-REWRITE AFTER I GOT SOME ASTRONOMICAL TIMES AND COULD NOT
###             FIND THE ERROR. SO, THIS CODE IS NOT PRETTY IN ANY WAY. BUT IT WORKS.
###             I HOPE I'LL HAVE TIME TO REWRITE IT ALL IT GOOD CODE-STYLE, BUT IF YOU SEE THIS,
###             THEN I HAD NO TIME FOR THAT (THANKS TO ALL THE OTHER SUBJECTS ON OUR FACULTY I HAVE TO DO)
###                                                                    ^ they are boring, tho
###

from random import random
import re
from math import log

def count_labels(labels):
    return {label: sum(1 for l in labels if l == label) for label in set(labels)}


def pullDict(texts, SIZE):
    D = []

    for i in range(SIZE):
        texts[i] = re.sub("(\.\.\.)", " <dots> ", texts[i])  # separate case: ...
        texts[i] = re.sub("<br />", "", texts[i])
        texts[i] = re.sub("(\.+)", ".", texts[i])  # any number of dots -> just one dot
        #temp = list(map(lambda x: x.lower(), texts[i].split()))
        texts[i] = " ".join(list(map(lambda x: x.lower(), texts[i].split())))  # lowercasing

        #texts[i] = re.sub("[\.\,\!\?\:\'\"]", "", texts[i])
        temp = texts[i].split()
        D += temp
        #texts[i] = " ".join(temp)
    return bi_gramm(list(D))

def process(texts):
    sents = 0
    SIZE = len(texts)

    for i in range(SIZE):
        #texts[i] = re.sub("(\.\.\.)", "3DOTS", texts[i]) # separate case: ...
        #texts[i] = re.sub("<br />", "", texts[i])
        #texts[i] = re.sub("(\.+)", ".", texts[i]) # any number of dots -> just one dot
        #texts[i] = " ".join(list(map(lambda x: x.lower(), texts[i].split()))) # lowercasing
        #texts[i] = list(filter(lambda x: len(x) > 0, texts[i].split('.')))
        sents += len(list(filter(lambda x: len(x) > 0, texts[i].split('.')))) # Returning sentence-separated lists
    #print(texts[:5], sents, SIZE)
    texts = (' '.join(texts)).split()
    #print(texts[:5], sents, SIZE)
    bi_gramm(texts)
    return texts, sents, SIZE

def process_2(texts):
    sents = 0
    SIZE = len(texts)

    for i in range(SIZE):
        texts[i] = re.sub("(\.\.\.)", "3DOTS", texts[i]) # separate case: ...
        texts[i] = re.sub("<br />", "", texts[i])
        texts[i] = re.sub("(\.+)", ".", texts[i]) # any number of dots -> just one dot
        texts[i] = list(map(lambda x: x.lower(), texts[i].split())) # lowercasing

    return texts

### Adding 3-gramms together with 2-gramms made
### the accuracy has risen even further
### but the Dict size skyrocketed...

### There's a mistake in the way i create 3-grams, but
### when i corrected it, the accuracy fell a bit.(so i left the wrong version)
def bi_gramm(words):
    gram = []
    for i_w in range(len(words) - 1):
        gram.append(words[i_w] + ' ' + words[i_w + 1])
    words+= gram

    for i_w in range(len(words) - 2):
        gram.append(words[i_w] + ' ' + words[i_w + 1] + ' ' + words[i_w + 2])
    words+= gram

    return words

def wfCount(text, Dict, min_cnt = 0):
    wf = {}
    for w in Dict:
        wf[w] = 0

    for word in text:
        if word in wf.keys():
            wf[word] += 1
        else:
            wf[word] = 1

    NUM_OF_WORDS = sum(list(wf.values()))
    
    for w in wf.keys():
        alpha = 1
        wf[w] += alpha
        wf[w] /= (NUM_OF_WORDS + alpha * len(Dict))

    return wf

def train(train_texts, train_labels):
    #print(set(train_labels))
    SIZE = len(train_texts)
    Dict = list(set(pullDict(train_texts, SIZE)))

    print(len(Dict), "<- Dict size")
    # separating neg & pos
    zipped = list(zip(train_texts, train_labels))
    negatives = [x for (x, _) in filter(lambda x: x[1] == "neg", zipped)]
    positives = [x for (x, _) in filter(lambda x: x[1] == "pos", zipped)]

    print(len(negatives))
    print(len(positives))

    pos, pos_sents, pos_SIZE = process(positives)
    neg, neg_sents, neg_SIZE = process(negatives)

    # 0 -> neg
    # 1 -> pos

    prior = [
             neg_SIZE / SIZE,
             pos_SIZE / SIZE
            ]


    pwf = wfCount(pos, Dict)
    nwf = wfCount(neg, Dict)

    return prior, pwf, nwf, Dict


#c = 0
def predict(txt, prior, pwf, nwf, Dict):
    #global c
    #c+= 1

    p_neg = log(prior[0])
    p_pos = log(prior[1])
    for w in txt:
        if w in nwf.keys() and w in pwf.keys():
            p_neg += log(nwf[w])
            p_pos += log(pwf[w])

    #print(c, "/25000")

    if p_pos > p_neg:
        return "pos"
    else:
        return "neg"






def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    res = [predict(bi_gramm(text), *params) for text in process_2(texts)]  # this dummy classifier returns random labels from p(label)
    print('Predicted labels counts:')
    #print(count_labels(res))
    return res

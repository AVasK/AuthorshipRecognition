# Text Processing:

import re
from nltk.stem import SnowballStemmer
import pymorphy2 
from nltk.tokenize import word_tokenize

def parse(text):
    # deleting all punctuation for now.
    sents = [t for t in re.split(r'[;!.?\n]', text) if t] # sentence tokenizing
    
    def word_nf(w):
        #morph = pymorphy2.MorphAnalyzer()
        stemmer = SnowballStemmer(language = 'russian')
        #return morph.parse(w)[0].normal_form
        return stemmer.stem(w)
    
    #stemmer = SnowballStemmer(language = 'russian')
    word_tokenize_1 = lambda s : [w for w in re.split(r'[\s,]', s) if w]
    word_tokenize_2 = word_tokenize # from NLTK
    tokenize_n_stem = lambda s : [word_nf(w) for w in word_tokenize_1(s)]
    sents = list(map(tokenize_n_stem, sents)) # words in sentences

    sents = ['. '.join(sent) for sent in sents]
    return sents
    

#print(re.split(r'[;!.?\n]', 'Do you really? Oh; Ok! N\never mind...'))
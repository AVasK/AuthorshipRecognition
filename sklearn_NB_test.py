from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(fit_prior = False)

from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vect = CountVectorizer(ngram_range = (1,2))
#vect = TfidfVectorizer()

from sklearn.model_selection import train_test_split

import re

from TextProcessor import parse

N_split = 5

AUTHORS = ['Толстой', 'Бунин', 'Гоголь']

def read_text(filename):
    with open(filename) as file:
        text = file.read()
        #sents = [t for t in re.split(r'[;!.?\n]', text) if t]
        sents = parse(text)
        
    return sents


def group_sents(sents, chunksize):
    chunked = []
    for i in range(0, len(sents) - chunksize, chunksize):
        chunked.append('. '.join(sents[i:i+chunksize]))
        
    return chunked
           
    
    
T = group_sents(read_text('Tolstoy.txt'), N_split)
B = group_sents(read_text('Boonin.txt'), N_split)
G = group_sents(read_text('Gogol.txt'), N_split)


X = T + B + G

assert len(X) == len(set(X))
y = [0] * len(T) + [1] * len(B) + [2] * len(G)

X = vect.fit_transform(X)
#print(vect.get_feature_names())

X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)


clf.fit(X, y)

"""
# simple test
Test_text = read_text('test.txt')

Test = vect.transform(Test_text)
print('\n> '.join([f"{t} : {s}" for t,s in zip(Test_text, clf.predict(Test))]))
"""

print(clf.score(X_test, y_test))

#q = input()
with open('tester.txt') as file:
    q = file.read()
q = parse(q)
#q = group_sents(q, 10)
q = '. '.join(q)
q = vect.transform([q])

auth = clf.predict(q)
#print(AUTHORS[auth[0]], ' : ', clf.predict_proba(q)[0][auth[0]])
print(auth)


if __name__ == 'test':
    Ts = read_text('Tolstoy.txt')
    Ks = read_text('king.txt')
    Gs = read_text('Gogol.txt')
    

    sent_len_avg = lambda sents : sum(len(s) for s in sents)/len(sents)
        
    print('Average length of sentences:\n', list(map(sent_len_avg, [Ts, Ks, Gs])))
    
    # calculate Divergence, Mean, etc.
    
    print(sent_len_avg(read_text('tester.txt')))

#! /bin/env python3

import xml.etree.ElementTree as ET
from gensim.models import Word2Vec, FastText
from nltk.corpus import stopwords
import numpy as np

right_wing = ['m', 'fp', 'c', 'all']
left_wing = ['s', 'v', 'mp']

class Model():
    def __init__(self, path):
        self.corpus = Corpus(path)
        self.model = None

    def make_model(self, conditions, model_name):
        texts = self.corpus.filter_texts(conditions)
        sentences = [self.corpus.words(text, True) for text in texts]
        sent = open('sentences.txt', 'w+')
        sent.close()
        model = Word2Vec(sentences, min_count=1, workers=5, sg=1)
        model.save(model_name)

        self.model = model

    def load_model(self, path):
        self.model = Word2Vec.load(path)

    def print_most_similar(self, word):
        if self.model is not None:
            try:
                for i, pair in enumerate(self.model.wv.most_similar(word, topn=10)):
                    print(str(i+1) + "\t" + pair[0] + "\t" + str(pair[1]))
            except KeyError:
                print("Word not in vocabulary")

class RIModel():
    def __init(self, path):
        self.corpus = Corpus(path)
        self.model = None



def year(s):
    """Converts a string representing a year to an integer

    This function assumes that the string is of the form YYYY(\w)?
    """
    if s.isdigit():
        return int(s)
    return int(s[0:-1])

class Corpus():
    def __init__(self, path):
        self.root = ET.parse(path)

    def get_text(self, text_id):
        words = []
        for text in self.root.findall('text'):
            if text.attrib['id'] == text_id:
                for word in text:
                    words.append(word.text)
        return ' '.join(words)

    def words(self, text, lemma=False, ignore_stopwords=True):
        words = []
        for word in text.findall('w'):
            #print(word.text)
            s = word.attrib['lemma'] if lemma else word.text
            #print(s)
            if lemma and word.attrib['lemma'] == "":
                continue
            else:
                  if ignore_stopwords and s.lower() not in stopwords.words('swedish_special'):
                    words.append(s)
        return words

    def filter_texts(self, conditions):
        texts = []
        for text in self.root.findall('text'):
            if all(cond(text.attrib[attr]) for attr, cond in conditions.items()):
                texts.append(text)
        return texts


if __name__ == '__main__':
    #corpus = Corpus('extracted.xml')
    #s = corpus.get_text('all-valmanifest-2006')

    # All documents pertaining to S and V before 1950
    #texts = corpus.filter_texts({'year': lambda t: year(t) <1950, 'party': lambda t: t in ['s', 'v']})
    #for text in texts:
    #    print (text.attrib['year'] + " " + text.attrib['party'])

    #print(' '.join(corpus.words(texts[0], lemma=False)))
    m = Model('extracted.xml')
    m.make_model({'year': lambda t: year(t) > 1960, 'party': lambda t: t in right_wing}, 'länge leve kapitalismen')
    #m.load_model("en bättre höger")
    m.print_most_similar('kultur')
    #u = m.model.wv['skatt']
    #v = m.model.wv['långsiktig']
    #print(u)
    #print(v)
    #print(u - v)
    #print(np.dot(u-v, u-v))

    #texts = corpus.filter_texts({'party': lambda t: t == 'm'})
    #print(' '.join(corpus.words(texts[1], lemma=False)))

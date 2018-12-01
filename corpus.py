#! /bin/env python3

import xml.etree.ElementTree as ET

right_wing = ['m', 'fp', 'c', 'all']
left_wing = ['s', 'v', 'mp']

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

    def words(self, text, lemma=False):
        words = []
        for word in text.findall('w'):
            #print(word.text)
            s = word.attrib['lemma'] if lemma else word.text
            #print(s)
            if lemma and word.attrib['lemma'] == "" :
                continue
            else:
                words.append(s)
        return words

    def filter_texts(self, conditions):
        texts = []
        for text in self.root.findall('text'):
            if all(cond(text.attrib[attr]) for attr, cond in conditions.items()):
                texts.append(text)
        return texts


if __name__ == '__main__':
    corpus = Corpus('extracted.xml')
    s = corpus.get_text('all-valmanifest-2006')
    
    # All documents pertaining to S and V before 1950
    texts = corpus.filter_texts({'year': lambda t: year(t) <1950, 'party': lambda t: t in ['s', 'v']})
    for text in texts:
        print (text.attrib['year'] + " " + text.attrib['party'])

    print(' '.join(corpus.words(texts[0], lemma=False)))


#! /bin/env python3

import xml.etree.ElementTree as ET

right_wing = ['m', 'fp', 'c', 'all']
left_wing = ['s', 'v', 'mp']

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


if __name__ == '__main__':
    corpus = Corpus('extracted.xml')
    print(corpus.get_text('v-partiprogram-1921'))
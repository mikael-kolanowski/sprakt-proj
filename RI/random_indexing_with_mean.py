import os
import argparse
import time
import string
import numpy as np
import tempfile
from nltk.corpus import stopwords
from halo import Halo
from itertools import combinations
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors


class RandomIndexing(object):
    def __init__(self, filenames, dimension=2000, non_zero=100, non_zero_values=[-1, 1], left_window_size=6, right_window_size=6):
        self.__sources = filenames
        self.vocab = set()
        self.__dim = dimension
        self.__non_zero = non_zero
        self.__non_zero_values = non_zero_values
        self.__lws = left_window_size
        self.__rws = right_window_size
        self.__cv = None
        self.__rv = None

    def clean_line(self, line):
        import string
        words = line.split()
        stopchars = '0123456789' + string.punctuation
        remove = str.maketrans('','',stopchars)
        clean_words = [word.translate(remove) for word in words]
        return list(filter(None, clean_words))


    def text_gen(self):
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def build_vocabulary(self):
        """
        Build vocabulary of words from the provided text files
        """
        linegen = self.text_gen()
        for line in linegen:
            for word in line:
                self.vocab.add(word)
        self.write_vocabulary()


    @property
    def vocabulary_size(self):
        return len(self.vocab)


    def create_word_vectors(self):
        """
        Create word embeddings using Random Indexing
        """
        self.__cv = {word:np.zeros(self.__dim) for word in self.vocab}
        self.__rv = {word:np.zeros(self.__dim) for word in self.vocab}
        for word in self.vocab:
            randindex = np.random.choice(self.__dim - 1, self.__non_zero, replace=False)
            for ind in randindex:
                self.__rv[word][ind] = np.random.choice(self.__non_zero_values)

        linegen = self.text_gen()
        for line in linegen:
            words = line
            for i, word in enumerate(line):
                cumul = np.zeros(self.__dim)
                for n in range(self.__lws):
                    if i - n - 1 >= 0:
                        cumul = (self.__rv[words[i - n - 1]]) / (n + 1) + cumul
                for n in range(self.__rws):
                    if i + n + 1 <= len(words) - 1:
                        cumul = (self.__rv[words[i + n + 1]]) / (n + 1) + cumul
                self.__cv[word] = self.__cv[word] + cumul
        pass


    def find_nearest(self, words, k=10, metric='cosine'):
        """
        Function returning k nearest neighbors for each word in `words`
        """
        vectors = list(self.__cv.values())
        knn = NearestNeighbors(n_neighbors=k+1, metric=metric)
        knn.fit(vectors)

        output = []
        for word in words:
            try:
                mini_output = []
                wordvec = self.get_word_vector(word)
                neighbors = knn.kneighbors([wordvec])
                for i in range(1,k+1):
                    mini_output.append((list(self.__cv.keys())[neighbors[1][0][i]],round(neighbors[0][0][i], 2)))
                output.append(mini_output)
            except ValueError:
                return []

        return output


    def get_word_vector(self, word):
        """
        Returns a trained vector for the word
        """
        if word in self.vocab:
            return self.__cv[word]
        return None


    def write_vocabulary(self):
        with tempfile.NamedTemporaryFile('w') as f:
            self.tempname = f.name
            for w in self.vocab:
                f.write('{}\n'.format(w))


    def train(self):
        """
        Main function call to train word embeddings
        """
        spinner = Halo(spinner='arrow3')
        spinner.start(text="Building vocabulary...")
        start = time.time()
        self.build_vocabulary()
        spinner.succeed(text="Built vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), self.vocabulary_size))

        spinner.start(text="Creating vectors using random indexing...")
        start = time.time()
        self.create_word_vectors()
        spinner.succeed("Created random indexing vectors in {}s.".format(round(time.time() - start, 2)))

        spinner.succeed(text="Successfully trained model!")

    def test_vectors(self):
        self.train()
        text = input('> ')
        while text != 'exit':
            print(self.get_word_vector(text))
            text = input('> ')


def compare_distance(word1, word2, ris):
    output = []
    for mdl in ris:
        v1 = mdl.get_word_vector(word1)
        v2 = mdl.get_word_vector(word2)
        output.append(cosine(v1, v2))
    return output

def automatic_testing(models):
    pop_size = 20
    test_results = {key: [[] for key in models.keys()] for key in models.keys()}
    common = set.intersection(*[model.vocab for model in models.values()])
    common = filter(lambda t: t is not None,
        [word if word not in stopwords.words('swedish') else None for word in common])

    sample = np.random.choice(list(common), pop_size)
    for word in sample:
        print(word)
        for model_name, model in models.items():
            nearest_word, distance = model.find_nearest([word], k=1)[0][0]
            for i, model in enumerate(models.values()):
                wv_nearest = model.get_word_vector(nearest_word)
                wv_test = model.get_word_vector(word)
                if wv_nearest is None:
                    continue
                test_distance = np.abs(cosine(wv_test, wv_nearest) - distance)
                if test_results[model_name][i] is []:
                    test_tesults[model_name][i] = [test_distance]
                else:
                    test_results[model_name][i].append(test_distance)


    tmp_storage = {}
    for model1_name in models.keys():
        for i, model2_name in enumerate(models.keys()):
            if model1_name != model2_name:
                if (model2_name, model1_name) in tmp_storage:
                    try:
                        print('Average distance between {} and {}: {}'.format(model1_name,
                            model2_name,
                            (tmp_storage[(model2_name, model1_name)] + np.mean(test_results[model1_name][i])))) / 2
                    except TypeError as err:
                        print(model1_name + " " + model2_name)
                else:
                    tmp_storage[(model1_name, model2_name)] = np.mean(test_results[model1_name][i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Indexing word embeddings')
    parser.add_argument('-fv', '--force-vocabulary', action='store_true', help='regenerate vocabulary')
    parser.add_argument('-c', '--cleaning', action='store_true', default=False)
    parser.add_argument('-co', '--cleaned_output', default='cleaned_example.txt', help='Output file name for the cleaned text')
    args = parser.parse_args()

    if args.force_vocabulary:
        os.remove('vocab.txt')

    if args.cleaning:
        ri = RandomIndexing(['example.txt'])
        with open(args.cleaned_output, 'w') as f:
            for part in ri.text_gen():
                f.write("{}\n".format(" ".join(part)))
    else:
        dir_name = "data"
        filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]
        ri_dict = {}
        for filename in filenames:
            ri_dict[filename] = RandomIndexing([filename])
            ri_dict[filename].train()
#            ri.test_vectors()
        print('Skriv ett ord:')
        text = input('> ')
        while text != '':
            text = text.split()
            if text[0] == 'compare':
                dist = compare_distance(text[1], text[2], ri_dict.values())
                for i, mdl in enumerate(ri_dict.keys()):
                    print('Distance between {} and {} in {}: {}'.format(text[1], text[2], mdl, dist[i]))
            elif text[0] == 'autotest':
                automatic_testing(ri_dict)
            else:
                for model_name, ri in ri_dict.items():
                    neighbors = ri.find_nearest(text)
                    if neighbors:
                        for w, n in zip(text, neighbors):
                            #print("Neighbors for {} in {}: {}".format(w, model_name, n))
                            print('Neighbors for {} in {}'.format(w, model_name))
                            for i, pair in enumerate(n):
                                print(str(i+1) + " " + str(pair[0]) + "\t\t\t" + str(pair[1]))
                        print('')
            text = input('> ')

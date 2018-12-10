import os
import argparse
import time
import string
import numpy as np
import tempfile
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
    pop_size = 500
    common = set.intersection(*[model.vocab for model in models.values()])
    distances = []
    sample = np.random.choice(list(common), pop_size)
    for i in range(int(pop_size / 2)):
        index1, index2 = np.random.randint(pop_size, size=2)
        distances.append(compare_distance(sample[index1], sample[index2], models.values()))
    for comb in combinations(models.keys(), r=2):
        distlist = []
        for distance in distances:
            name_to_index = {name:i for i, name in enumerate(models.keys())}
            distlist.append(abs(distance[name_to_index[comb[0]]] - distance[name_to_index[comb[1]]]))
        print('Average distance between {} and {}: {}'.format(comb[0], comb[1], np.mean(distlist)))

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

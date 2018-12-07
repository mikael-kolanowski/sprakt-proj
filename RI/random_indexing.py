import os
import argparse
import time
import string
import numpy as np
import tempfile
from halo import Halo
from sklearn.neighbors import NearestNeighbors


class RandomIndexing(object):
    def __init__(self, filenames, dimension=2000, non_zero=100, non_zero_values=[-1, 1], left_window_size=3, right_window_size=3):
        self.__sources = filenames
        self.__vocab = set()
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
                self.__vocab.add(word)
        self.write_vocabulary()


    @property
    def vocabulary_size(self):
        return len(self.__vocab)


    def create_word_vectors(self):
        """
        Create word embeddings using Random Indexing
        """
        self.__cv = {word:np.zeros(self.__dim) for word in self.__vocab}
        self.__rv = {word:np.zeros(self.__dim) for word in self.__vocab}
        for word in self.__vocab:
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
                        cumul = np.add(self.__rv[words[i - n - 1]], cumul)
                for n in range(self.__rws):
                    if i + n + 1 <= len(words) - 1:
                        cumul = np.add(self.__rv[words[i + n + 1]], cumul)
                self.__cv[word] = np.add(self.__cv[word], cumul)
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

#            vds = []
#            for word in self.__vocab:
#                vds.append((word,distance))

        return output


    def get_word_vector(self, word):
        """
        Returns a trained vector for the word
        """
        if word in self.__vocab:
            return self.__cv[word]
        return None


    # def vocab_exists(self):
    #     return os.path.exists(self.tempname)
    #
    #
    # def read_vocabulary(self):
    #     vocab_exists = self.vocab_exists()
    #     if vocab_exists:
    #         with open(self.tempname) as f:
    #             for line in f:
    #                 self.__vocab.add(line.strip())
    #     self.__i2w = list(self.__vocab)
    #     return vocab_exists


    def write_vocabulary(self):
        with tempfile.NamedTemporaryFile('w') as f:
            self.tempname = f.name
            for w in self.__vocab:
                f.write('{}\n'.format(w))


    def train(self):
        """
        Main function call to train word embeddings
        """
        spinner = Halo(spinner='arrow3')

        # if self.vocab_exists():
        #     spinner.start(text="Reading vocabulary...")
        #     start = time.time()
        #     ri.read_vocabulary()
        #     spinner.succeed(text="Read vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        # else:
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
            for model_name, ri in ri_dict.items():
                neighbors = ri.find_nearest(text)
                if neighbors:
                    for w, n in zip(text, neighbors):
                        #print("Neighbors for {} in {}: {}".format(w, model_name, n))
                        print('Neighbors for {} in {}'.format(w, model_name))
                        for i, pair in enumerate(n):
                            print(str(i+1) + " " + str(pair[0]) + "\t\t" + str(pair[1]))
                    print('\n')
            text = input('> ')

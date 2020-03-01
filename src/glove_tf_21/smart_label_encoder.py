from glove_tf_21.utils import safe_key_extractor
from collections import Counter
from itertools import islice
from p_tqdm import p_uimap


class SmartLabelEncoder(object):

    def __init__(self, min_occurrence=3, max_features=10000, unk_token="UNK", n_jobs=-1):

        """
        Inspires by the LabelEncoder from sklearn, replaces words with tokens
        2 differences:
        - works on text files directly
        - parallellized
        - will replace out of vocabulary tokens by an UNK token index = 0

        :param min_occurrence:number of times a value has to appear to be considered
        :param max_features: max number of features to keep
        :param unk_token: value to replace the out of vocabulary values by
        :param n_jobs: number of parallel processes to run on
        """

        self.min_occurrence = min_occurrence
        self.max_features = max_features
        self.unk_token = unk_token

        self.n_jobs = n_jobs

        self.counter = None
        self.classes_ = list()

        self.val2ix = dict()
        self.ix2val = dict()

        self.features_n = None

    @staticmethod
    def count_in_file(file_path):
        """
        Count the frequency of tokens in a file (tokens separated by space)
        :param file_path: path to the text file
        :return: collections.Counter object
        """
        counter = Counter()
        with open(file_path, 'r') as infile:
            try:
                while True:
                    lines_gen = islice(infile, 10000)
                    counter.update(next(lines_gen).split())
            except StopIteration:
                pass
        return counter

    def fit(self, X):
        """

        Creates the vocabulary

        :param X: can be an array, a list or a list of files
        :return: None
        """

        # 1. Create counter
        # 1.a Create counter from file
        if isinstance(X, str):
            self.counter = self.count_in_file(X)

        # 1.b Create counter from multiple files in parallel
        elif isinstance(X, list) and isinstance(X[0], str):
            self.counter = Counter()
            counter_iterator = p_uimap(self.count_in_file, X)

            for temp_counter in counter_iterator:
                self.counter += temp_counter

        #  1.c Use counter Object directly
        else:
            self.counter = Counter(X)

        # 2. Remove UNK token if present in the vocabulary (it will be added at the beginning of the vocabulary)
        if self.unk_token in self.counter:
            self.counter.pop(self.unk_token)

        # 3. Keep only tokens that appeared more than min_occurrence times
        self.counter = Counter(
            {k: c for k, c in self.counter.most_common(self.max_features - 1)
             if c >= self.min_occurrence}
        )

        # 4. Instantiate dictionaries val2ix, ix2val and list of classes
        self.val2ix[self.unk_token] = 0
        self.ix2val[0] = self.unk_token
        self.classes_ = [self.unk_token]

        # 5. Fill those dictionaries with the vocabulary and indexes
        for ix, (val, _) in enumerate(self.counter.most_common()):

            self.val2ix[val] = ix + 1
            self.ix2val[ix + 1] = val
            self.classes_.append(val)

        self.features_n = len(self.classes_)

    def val_to_index(self, tok_key):
        """
        Returns index of token, if token is out of vocabulary, return the index of the UNK token

        :param tok_key: token value (str)
        :return: ix
        """
        return safe_key_extractor(self.val2ix, tok_key, default_val=0)

    def index_to_val(self, ix_key):
        """
        Returns token of index, if index is out of vocabulary, return the UNK token

        :param ix_key: index (int)
        :return: token (str)
        """
        return safe_key_extractor(self.ix2val, ix_key, default_val=self.unk_token)

    @staticmethod
    def process_array_or_str(X, func, elem_type):
        """

        :param X:
        :param func:
        :param elem_type:
        :return:
        """

        if len(X) > 0:
            if isinstance(X[0], elem_type):
                return list(map(func, X))
            else:
                return [list(map(func, sequence)) for sequence in X]
        else:
            return []

    def transform(self, X):
        """
        change an array of tokens to an array of indexes

        :param X: array or list of tokens
        :return: array of indexes
        """
        return self.process_array_or_str(X, self.val_to_index, str)

    def inverse_transform(self, X):
        """
        change an array of indexes to an array of tokens

        :param X: array or list of indexes
        :return: array of tokens
        """
        return self.process_array_or_str(X, self.index_to_val, int)

    def fit_transform(self, X):
        """
        Fits vocabulary on an array of tokens (or a list of files)

        :param X:  can be an array, a list or a list of files
        :return: array of tokens
        """
        self.fit(X=X)
        return self.transform(X=X)

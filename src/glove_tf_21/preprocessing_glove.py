from glove_tf_21.smart_label_encoder import SmartLabelEncoder
from glove_tf_21.utils import get_files_in_folder
from multiprocessing import cpu_count
from scipy.sparse import coo_matrix
from itertools import islice
from p_tqdm import p_uimap
from time import time
import numpy as np
import tempfile
import shutil
import logging
import os


class PreprocessingGlove(object):

    def __init__(self, data_path, window_size=5, unk_token="UNK", max_features=10000, min_occurrence=5):

        """
        Pre-processing function for the Glove Algorithm,
        Takes a list of text files (one sentence per line,tokens separated by spaces) as an input
        Outputs the sparse co-occurrence matrix: list of rows,list of columns and list of values

        :param data_path: path to the files
        :param window_size: size of the window to consider for the co-occurrence
        :param unk_token: Unknown token to replace words out of vocabulary
        :param max_features: max number of words in the vocabulary (unknown token is counted)
        :param min_occurrence: min occurrence of a word to consider it
        """

        self.data_path = data_path
        self.window_size = window_size
        self.unk_token = unk_token
        self.max_features = max_features
        self.min_occurence = min_occurrence

        self.files = get_files_in_folder(self.data_path)
        self.files_n = len(self.files)

        # Split dataset in multiple files to make process faster
        if (self.files_n < cpu_count()) and (os.path.getsize(self.files[0]) > 1E6):
            self.temp_dir = tempfile.mkdtemp()
            for file in self.files:
                os.system(
                    f"split -l 100000 -d {os.path.join(self.data_path, file)} " 
                    f"{os.path.join(self.temp_dir, os.path.basename(file))}_tmp_"
                )
            self.files_to_process = [os.path.join(self.temp_dir, file) for file in os.listdir(self.temp_dir)]
            logging.info(f"temp folder {self.temp_dir} created")
        else:
            self.temp_dir = None
            self.files_to_process = self.files

        # Create a label encoder
        self.sle = SmartLabelEncoder(max_features=self.max_features, min_occurrence=self.min_occurence)
        self.vocab_size = None

    def cooc_count(self, cooc_dict, sequence_ix):
        """

        :param cooc_dict: co-occurrence dict (keys are a tuple of indexes)
        :param sequence_ix: list of indexes
        :return:
        """
        for elem_pos, elem_ix in enumerate(sequence_ix):
            for dist in range(1, min(self.window_size + 1, len(sequence_ix) - elem_pos)):
                first_id, second_id = sorted([elem_ix, sequence_ix[elem_pos + dist]])
                if (first_id, second_id) in cooc_dict.keys():
                    cooc_dict[(first_id, second_id)] += 1.0 / dist
                else:
                    cooc_dict[(first_id, second_id)] = 1.0 / dist
        return cooc_dict

    def get_labels(self):
        """

        :return: list of tokens, ordered
        """
        return self.sle.classes_

    def cooc_dict_to_sparse(self, cooc_dict):

        """
        Convert dict key: (row_ix, col_ix) value: co-occurrence
        :param cooc_dict: dict
        :return: sparse matrix (coo)
        """

        array_vals = np.array(list(cooc_dict.values()))
        array_keys = np.array(list(cooc_dict.keys()))
        cooc = coo_matrix((array_vals, (array_keys[:, 0], array_keys[:, 1])), shape=(self.vocab_size, self.vocab_size),
                          dtype=np.float32)

        del cooc_dict
        del array_keys
        del array_vals

        return cooc

    @staticmethod
    def glove_formatter(cooc):
        """
        Separate rox indexes, col indexes and values in a sparse matrix
        :param cooc: sparse coo matrix
        :return: 3 arrays
        """
        cooc_rows, cooc_cols = cooc.nonzero()
        return cooc_rows, cooc_cols, cooc.data

    def get_cooc_mat(self, file_name):
        """
        Generate the sparse matrix of co-occurrence from a text file

        :param file_name: path to the file
        :return: co-occurrence matrix
        """
        partial_cooc_dict = dict()
        # Open the file
        with open(file_name, 'r') as infile:
            try:
                while True:
                    lines_gen = islice(infile, 10000)

                    # File to list of list of indexes (one list per sentence)
                    ix_sentence = self.sle.transform(next(lines_gen).split())

                    # Create co-occurrence matrix form list of list of indexes
                    partial_cooc_dict = self.cooc_count(partial_cooc_dict, ix_sentence)
            except:
                pass

        if len(partial_cooc_dict):
            # Change to sparse matrix
            partial_cooc = self.cooc_dict_to_sparse(partial_cooc_dict)
        else:
            # If empty data
            partial_cooc = coo_matrix((self.sle.features_n, self.sle.features_n))
        return partial_cooc

    def __call__(self):

        # 1. Fit the vocabulary (read all the text and decide which tokens to keep in the vocabulary)
        start_time = time()
        self.sle.fit(self.files_to_process)
        self.vocab_size = self.sle.features_n
        logging.info(f" - labels fit done {time() - start_time: 2f} seconds")

        # 2. Parse all the files and generate co-occurrence matrices for each file in a sparse format
        start_time = time()
        cooc_iterator = p_uimap(self.get_cooc_mat, self.files_to_process)

        cooc = coo_matrix((self.sle.features_n, self.sle.features_n))
        for partial_cooc in cooc_iterator:
            # 3. add all the co-occurrence matrices together
            cooc += partial_cooc
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir)
            logging.info(f"temp folder {self.temp_dir} deleted")
        logging.info(f" - cooc dict computed {time() - start_time: 2f} seconds")

        # 4. Format the sparse matrix to get indices and values separated
        start_time = time()
        cooc_rows, cooc_cols, cooc_data = self.glove_formatter(cooc)
        logging.info(f" - output formated {time() - start_time: 2f} seconds")
        return cooc_rows, cooc_cols, cooc_data, cooc

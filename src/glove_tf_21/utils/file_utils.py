import os


def flatten_lists_of_lists(lists_of_lists):
    """
    Flatten list of lists into a list
    ex:
        [[1, 2], [3]] => [1, 2, 3]

    :param lists_of_lists: list(list(elems))
    :return: list(elems)
    """

    return [elem for sublist in lists_of_lists for elem in sublist]


def safe_key_extractor(dico, key, default_val):
    """
    Get the value from a dictionary based on a key,

    If the key is not present, return the default value

    :param dico: dict
    :param key: key present or not in the dictionary
    :param default_val: default value to return
    :return: value or default_val
    """
    try:
        return dico[key]
    except KeyError:
        return default_val


def get_files_in_folder(path):
    """
    Get the paths of files in a folder
    If path is already a file, return a value containing this value

    :param path: str
    :return: list of str
    """
    if os.path.isfile(path):
        return [path]
    else:
        return [os.path.join(path, file_name) for file_name in os.listdir(path)]


def get_train_files(data_path):
    """
    Get the files in a folder like previous function, iif the file name contains 'train'

    :param data_path: str
    :return: list of str
    """
    return [file for file in get_files_in_folder(data_path) if 'train' in file]


def get_val_files(data_path):
    """
    Get the files in a folder like previous function, iif the file name contains 'val'

    :param data_path: str
    :return: list of str
    """
    return [file for file in get_files_in_folder(data_path) if 'val' in file]


def save_labels(labels, path):
    """
    Save a list of labels into a text file, one element per line

    :param labels: list of labels
    :param path: str file path
    :return: None
    """
    with open(path, "w") as f:
        for label in labels[:-1]:
            f.write(f"{label}\n")
        f.write(str(labels[-1]))


def load_file(file_path):
    """
    load file and return a list of elements, one per line

    :param file_path: str
    :return: list of str
    """
    return open(file_path, "r").read().split("\n")


def load_corpus_file(file_path):

    """
    Load text file and return list of list of str

    :param file_path: str
    :return: list of str
    """

    return [
        line.split() for line in load_file(file_path)
    ]


def load_size(file_path):
    return int(load_file(file_path)[0])


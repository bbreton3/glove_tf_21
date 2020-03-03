[![Build Status](https://travis-ci.org/bbreton3/glove_tf_21.svg?branch=master)](https://travis-ci.org/bbreton3/glove_tf_21)
[![Coverage Status](https://coveralls.io/repos/github/bbreton3/glove_tf_21/badge.svg?branch=master)](https://coveralls.io/github/bbreton3/glove_tf_21?branch=master)


# Glove tf 2.1

The goal of this project is to re-implement the Glove embeddings algorithm on tensorflow 2.1


## Installation:

**NB**: This library has been tested on **python 3.6** and **python 3.7**
1. Clone the repository: `git clone git@github.com:bbreton3/glove_tf_21.git`
2. Move to the repository folder: `cd glove_tf_21`
3. Install the requirements:
    - **without GPU** support: `pip install -r requirements.txt`
    - **with GPU** support: (you must have CUDA 10.0 or 10.1 installed and configured) `pip install -r requirements-gpu.txt`
4. Install the library `pip install -e .`


## Pre-processing

To train the glove algorithm, your text corpus has to be pre-processed, the corpus has to be saved in a `.txt` file or a
bunch of `.txt` files (one sentence per line) saved in the same directory. 

The pre-processing will go through the whole corpus, save the `--max_features` 
most common words as a vocabulary, and then build a co-occurrence matrix containing the frequency of apparition of the 
words together in a sentence. The co-occurrence matrix **M** is of dimension **(max_features, max_features)**,

**M(i, j)** is a coefficient liked to the number of times where the i-st word and the j-st word appeared together in a sentence.

Because most words never appear together, most of the coefficients are 0 and the matrix is saved as a sparse matrix.

The sparse matrix is then formatted as 3 list of elements (type coo): row_index, col_index, matrix_element
which ars then saved as together, element by element in a **tfrecords** file that can be used to train the algorithm.

### Ex:

```bash
run_preprocessing.py \
    --data_path /your/data/path/ \
    --output_path /tfrecords/path/ \
    --max_features 20000 \
    --val_size 0.05 \
    --tf_split 12 
```

- `--data_path` path to the .txt file or the folder containing the .txt files
- `--output_path` path where you want to store the tfrecords
- `--max_features` size of the vocabulary to consider the UNK token is counted in this vocabulary size
- `--val_size` size of the corpus to keep for validation
- `--tf_split` number of tfrecords files, default is number of cpu cores

## The Model

The GLOVE Model was coded using the `tf.keras.Model` class and implementing by hand the `call` method which corresponds 
to a forward path in the algorithm training. 
The Model could also be implemented using the tf.keras.Sequential api
The input of the `call` method corresponds to the **row_index** and **col_index** of the co-occurrence matrix, they have 
to be concatenated to pass a single input to the model during training.
The glove loss is saved as a method of the model,but it is a simple function and could be saved elsewhere.

## Training

Once the tfrecords are saved, you can launch the training usin the scripts `run_training_keras.py` or `run_training.py`
The differences are:
- `run_training_keras.py` uses the tf.keras.Model.fit method that is easier to use and allows the use of Callbacks
- `run_training.py` uses a custom loop and `tf.GradientTape` method to compute the gradient after iteration and 
back-propagate this way of training is not as user-friendly as the other one, but it allows more flexibility if you wish 
to train the algorithm in a non-standard way.

### Ex:
```bash
python run_training_keras.py
        --data_path /tfrecords/path/
        --training_name my_training_v1
        --dim 300
        --batch_size 2048
        --epochs_number 50
        --initial_epoch 0
        --save_embeddings_every_epoch 5
```
The same options are available in the `run_training.py` script except for `--save_embeddings_every_epoch`

- `--data_path` path to the tfrecords files
- `--training_name` path where you want to store output model in the summaries folder
- `--dim` dimension of the embeddings
- `--batch_size` batch size (the model is very small, default is 10000)
- `--epochs_number` epoch number
- `--initial_epoch` "initial epoch number
- `--save_embeddings_every_epoch` number of epochs between saving the embeddings in numpy and in the tensorboard

You can stop the training and restart with the same *training name* the training will restart at the beginning of the 
last epoch.

### Convergence example:

The training time and convergence rate are the same with both training script.

## Tensorboard

to launch the `tensorboard` simply use the command `tensorboard --logdir=summaries/training_name` from the root of the 
project.

## Embeddings

The embeddings are saved as numpy arrays in the directory **summaries/yout_training_name/embeddings/epoch_n**
You will find the: **context embeddings**, **target embeddings** as well as **combined embeddings**
which is the average of the two former.

The embeddings are of dimension (max_features, dimension).
They can be imported in numpy and be used as a basis for another algorithm.

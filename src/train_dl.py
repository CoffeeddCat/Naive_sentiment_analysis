import tensorflow as tf
import numpy as np
import math
from loader import Loader
from config import *
from model import Network

if __name__ == '__main__':

    loader_pos = Loader(FILE_LOC_POS)
    loader_neg = Loader(FILE_LOC_NEG)

    network = Network("training")

    # for test
    test_data = {
        'words': np.zeros((10, Embedding_dim)),
        'tags': np.zeros((10, 1))
    }

    
import numpy as np
import wdbedding
from loader import Loader
from config import *
from model import Network
from tqdm import tqdm

if __name__ == '__main__':

    loader_pos = Loader(FILE_LOC_POS)
    loader_neg = Loader(FILE_LOC_NEG)

    network = Network("training")


    # for test
    """test_data = {
        'words': np.zeros((10, Embedding_dim)),
        'tags': np.zeros((10, 1))
    }"""

    word_vec_model = wdbedding.load_word2vec_model(EN)

    episode = 0
    for episode in tqdm(range(Learning_episodes)):
        data_pos = loader_pos.sample(Batch_size)
        data_neg = loader_neg.sample(Batch_size)
        target_pos = np.array([1 for i in range(Batch_size)])
        target_neg = np.array([0 for i in range(Batch_size)])
        train_data = {}
        train_data['words'] = []
        for sentence in (data_pos + data_neg):
            train_data['words'].append(wdbedding.embedding(word_vec_model, sentence, EN))
        train_data['tags'] = np.concatenate((target_pos, target_neg), axis=0)
        train_data['words'] = np.reshape(train_data['words'], (-1, 128, 100, 1))
        network.train(train_data)
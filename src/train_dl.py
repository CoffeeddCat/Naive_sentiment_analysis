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

    word_vec_model = wdbedding.load_word2vec_model(EN)

    episode = 0
    for episode in tqdm(range(Learning_episodes)):
        data_pos = loader_pos.sample(Batch_size)
        data_neg = loader_neg.sample(Batch_size)
        target_pos = np.array([1.0 for i in range(Batch_size)])
        target_neg = np.array([0.0 for i in range(Batch_size)])
        train_data = {}
        train_data['words'] = []
        for sentence in (data_pos + data_neg):
            train_data['words'].append(wdbedding.embedding(word_vec_model, sentence, EN))
        train_data['tags'] = np.reshape(np.concatenate((target_pos, target_neg), axis=0), (-1,1))
        # print(train_data['tags'])
        train_data['words'] = np.reshape(train_data['words'], (-1, Max_sentence_length, Embedding_dim, 1))
        loss = network.train(train_data)
        if episode % 100 ==0:
            print("now loss %f" % (loss))

    if TEST:
        data_pos = loader_pos.sample(Test_size)
        data_neg = loader_neg.sample(Test_size)
        target_pos = [1 for i in range(Test_size)]
        target_neg = [0 for i in range(Test_size)]
        target = target_pos + target_neg
        test_data = []
        for sentence in (data_pos + data_neg):
            test_data.append(wdbedding.embedding(word_vec_model, sentence, EN))

        test_data = np.reshape(test_data, (-1, Max_sentence_length, Embedding_dim, 1))
        result = network.get_result(test_data)
        correct = 0
        for index in range(len(result[0])):
            if result[0][index] > 0.5 and target[index] == 1 or result[0][index] < 0.5 and target[index] == 0:
                correct = correct + 1

        print(correct/(2*Test_size))
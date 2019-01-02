import numpy as np
import wdbedding
from loader import Loader
from config import *
from model import Network
import copy
import random
from tqdm import tqdm

if __name__ == '__main__':

    loader_pos = Loader(FILE_LOC_POS)
    loader_neg = Loader(FILE_LOC_NEG)
    # init the network
    network = Network("training")
    # init the embedding model
    word_vec_model = wdbedding.load_word2vec_model(EN)
    print("word2vec model_loaded")

    episode = 0

    data_pos = []
    data_neg = []
    # read in the training data
    for sentence in loader_pos.sentences:
        data_pos.append([np.array([1.0]),copy.deepcopy(sentence)])

    for sentence in loader_neg.sentences:
        data_neg.append([np.array([0.0]),copy.deepcopy(sentence)])

    data = copy.deepcopy(data_pos + data_neg)
    random.shuffle(data)

    for episode in range(Learning_episodes):
        # init in the step.
        loss = 0
        total_loss = 0
        random.shuffle(data)
        if TRAIN:
          for index in tqdm(range(len(data))):
              episode = 0
              # init the data feed in.
              training_data = {}
              training_data['words'] = []
              training_data['tags'] = []
              for i in range(1):
                  training_data['words'].append(wdbedding.embedding(word_vec_model, data[(index+i)%len(data)][1], EN))
                  training_data['tags'].append(data[(index+i)%len(data)][0])
              training_data['words'] = np.reshape(training_data['words'], (-1, Max_sentence_length, Embedding_dim, 1))
              training_data['tags'] = np.reshape(training_data['tags'], (-1,1))
              # get the loss.
              loss = network.train(training_data)
              total_loss = total_loss + loss
          print("now training step:%d, average loss: %f" % (episode, total_loss/len(data)))


        if TEST:
            # testing part.
            data_pos = loader_pos.sample_testing_set(Test_size)
            data_neg = loader_neg.sample_testing_set(Test_size)
            target_pos = [1 for i in range(len(data_pos))]
            target_neg = [0 for i in range(len(data_neg))]
            target = target_pos + target_neg
            test_data = []
            for sentence in (data_pos + data_neg):
                test_data.append(wdbedding.embedding(word_vec_model, sentence, EN))

            test_data = np.reshape(test_data, (-1, Max_sentence_length, Embedding_dim, 1))
            result = network.get_result(test_data)
            correct = 0
            correct_pos = 0
            correct_neg = 0
            for index in range(len(result[0])):
                if result[0][index] > 0.5 and target[index] == 1:
                    correct_pos = correct_pos + 1
                elif result[0][index] < 0.5 and target[index] == 0:
                    correct_neg = correct_neg + 1
            correct = correct_pos + correct_neg

            # output the performance on the testing set.
            print("pos correct:%d, neg correct: %d" % (correct_pos, correct_neg))
            print("correctness: %f"% (correct/(len(data_pos)+len(data_neg))))

from config import *
from model import Network
import numpy as np
import wdbedding
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

if __name__ == '__main__':

    network = Network("training")

    word_vec_model = wdbedding.load_word2vec_model(EN)
    print("word2vec model_loaded.")

    input_file_path = '../data/task2_input_en.xml'
    xmltree = ET.parse(input_file_path)
    xmlroot = xmltree.getroot()

    for review in xmlroot:
        txt = review.text
        if txt[-1] == '\n':
            txt = txt[:-1]

        print(txt)

        test_data = []
        test_data.append(wdbedding.embedding(word_vec_model, txt, EN))

        test_data = np.reshape(test_data, (-1, Max_sentence_length, Embedding_dim, 1))
        result = network.get_result(test_data)

        print(result)

        if (result[0][0]>0.5):
            review.set("polarity","1")

    output_file_path = '../data/task2_input_en.xml'
    xmltree.write(output_file_path, encoding="utf-8")

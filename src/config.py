# about the network
"""
some of the parameter is useless
"""
Embedding_dim = 100
Max_sentence_length = 128
Learning_rate = 1e-5
Model_load = False
CNN_Filters = 1
CNN_Kernel_size = 1
CNN_Strides = 1
DNN_Shape = [128, 32, 1]
Model_load_path = 1

# about the loader
FILE_LOC_NEG = "../data/sample.negative.cn.txt"
FILE_LOC_POS = "../data/sample.positive.cn.txt"
Loader_divided = False # the loader preprocess the sentences or not

# language setting
LANG = "CN"
CN = 0
EN = 1

# about the word embedding
Word_Embedding_Dir = '../word_embedding'

Puncts="?()。|？|！!|\n@*&#$%^_-+={}|\;:'\"<>，/ "

# about training
Every_steps_save = 2000
Learning_episodes = 100000
Batch_size = 20
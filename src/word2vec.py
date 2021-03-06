import gensim
"""
this code is for experiment. we don't use in our model.
"""
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
                
if __name__ == '__main__':
    sentences = MySentences('1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled')
    model = gensim.models.Word2Vec(sentences)
    model.save('en_word2vec.txt')
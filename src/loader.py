from config import *
import jieba
import copy
import random

class Loader:

    def __init__(self, file_path):
        self.file = open(file_path, mode='r', encoding="UTF-8")
        self.total_words = 0
        self.string = self.file.read()
        self.intab = ".!@-(),?;:\""
        self.outtab = "&&&&&&&&&&&"
        self.trans = str.maketrans(self.intab, self.outtab)
        self.initialize_sentences()
        self.take_apart()
        self.dict = {}
        self.initialize_dict()
        print(len(self.sentences))
        print(len(self.testing_sentences))

    def initialize_sentences(self):
        self.string = self.string.replace('\n', '')
        self.sentences_temp = self.string.split("</review>")
        self.sentences_temp.pop()  # delete the last blank sentence
        self.sentences = []

        if Loader_divided:
            # if the sentence is divided in the loader
            if LANG == "CN":
                move_ = ['《', '》', '！', '？', '，', '~', '`', ' ', '。',
                         '“', "”", '!', ',', '...', '；', '..', '.', ',', '．']
                for sentence in self.sentences_temp:
                    index = sentence.find(">")
                    sentence_temp = sentence[index + 1:]
                    sentence_temp = jieba.lcut(sentence_temp)
                    for x in move_:
                        flag = True
                        try:
                            while flag:
                                sentence_temp.remove(x)
                        except ValueError:
                            flag = False
                    self.sentences.append(copy.deepcopy(sentence_temp))
            elif LANG == "EN":
                for sentence in self.sentences_temp:
                    index = sentence.find(">")
                    sentence_temp = sentence[index + 1:]
                    sentence_temp = copy.deepcopy(
                        sentence_temp.translate(self.trans))
                    sentence_temp = sentence_temp.replace('&', '')
                    sentence_temp = sentence_temp.split(" ")
                    while True:
                        try:
                            sentence_temp.remove("")
                            sentence_temp.remove('')
                        except ValueError:
                            break
                    sentence_temp_ = []
                    for word in sentence_temp:
                        sentence_temp_.append(word.lower())
                    self.total_words = self.total_words + len(sentence_temp_)
                    self.sentences.append(copy.deepcopy(sentence_temp_))
            print(self.sentences)
        else:
            # if the sentence don't need be divided in the loader
            for sentence in self.sentences_temp:
                index = sentence.find(">")
                sentence_temp = sentence[index + 1:]
                self.sentences.append(copy.deepcopy(sentence_temp))


    def initialize_dict(self):
        # here is for the Naive Bayes to establish a dict.
        for sentence in self.sentences:
            s = set()
            for word in sentence:
                if not word in s:
                    try:
                        if self.dict[word] > 0:
                            self.dict[word] = self.dict[word] + 1
                    except KeyError:
                        self.dict[word] = 1
                    s.add(word)
        self.dict_len = len(self.dict)

    def sample(self, batch_num):
        # sample some sentences
        return random.sample(self.sentences, batch_num)

    def sample_testing_set(self, batch_num):
        # return all the testing set.
        return self.testing_sentences

    def take_apart(self):
        # divide the training set and the testing set.
        random.shuffle(self.sentences)
        index = int(len(self.sentences) * Testing_set_percent)
        self.testing_sentences = copy.deepcopy(self.sentences[:index])
        self.sentences = copy.deepcopy(self.sentences[index:])

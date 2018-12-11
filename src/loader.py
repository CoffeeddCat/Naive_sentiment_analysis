from config import *


class loader:

    def __init__(self):
        self.file = open(FILE_LOC, mode='r', encoding="UTF-8")
        self.string = self.file.read()
        self.initialize_sentences()

        print(self.sentences)
        # print(self.string)

    def initialize_sentences(self):
        self.string = self.string.replace("\n", "")
        self.sentences_temp = self.string.split("</review>")
        self.sentences_temp.pop()  # delete the last blank sentence
        self.sentences = []
        for sentence in self.sentences_temp:
            index = sentence.find(">")
            sentence_temp = sentence[index + 1:]


# for test
loader()

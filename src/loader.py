from config import *
import jieba
import copy


class Loader:

    def __init__(self, file_path):
        self.file = open(file_path, mode='r', encoding="UTF-8")
        self.total_words = 0
        self.string = self.file.read()
        self.intab = ".!@-(),?;:\""
        self.outtab = "&&&&&&&&&&&"
        self.trans = str.maketrans(self.intab, self.outtab)
        self.initialize_sentences()
        self.dict = {}
        self.initialize_dict()
        """for s in self.sentences:
            # s = s.translate(self.trans)
            print(s)
            print('\n')"""

    def initialize_sentences(self):
        self.string = self.string.replace('\n', '')
        self.sentences_temp = self.string.split("</review>")
        self.sentences_temp.pop()  # delete the last blank sentence
        self.sentences = []
        if LANG == "CN":
            move_ = ['《', '》', '！', '？', '，', '~', '`', ' ', '。',
                     '“', "”", '!', ',', '...', '；', '..', '.', ',', '．']
            for sentence in self.sentences_temp:
                index = sentence.find(">")
                sentence_temp = sentence[index + 1:]
                # sentence_temp = re.sub(r, '', sentence_temp)
                sentence_temp = jieba.lcut(sentence_temp)
                for x in move_:
                    flag = True
                    try:
                        while flag:
                            sentence_temp.remove(x)
                            # print(2333333)
                    except ValueError:
                        flag = False
                # print(sentence_temp)
                self.sentences.append(copy.deepcopy(sentence_temp))
        elif LANG == "EN":
            for sentence in self.sentences_temp:
                index = sentence.find(">")
                sentence_temp = sentence[index + 1:]
                sentence_temp = copy.deepcopy(
                    sentence_temp.translate(self.trans))
                sentence_temp = sentence_temp.replace('&', '')
                # print(sentence_temp)
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

    def initialize_dict(self):
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
        # print(self.dict)

# for test
# Loader(FILE_LOC_POS)

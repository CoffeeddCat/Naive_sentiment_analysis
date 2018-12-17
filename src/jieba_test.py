import jieba
import copy

file_path = "../data/sample.negative.cn.txt"
file = open(file_path, mode='r', encoding="UTF-8")
string = file.read()

move_ = ['《', '》', '！', '？', '，', '~', '`', ' ', '。',
         '“', "”", '!', ',', '...', '；', '..', '.', ',', '．']

string = string.replace('\n', '')
sentences_temp = string.split("</review>")
sentences_temp.pop()
sentences = []
for sentence in sentences_temp:
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
    print(sentence_temp)
    sentences.append(copy.deepcopy(sentence_temp))

# print(sentences)

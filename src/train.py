from loader import Loader
from collections import Counter
from config import *
import copy
import math

"""
here is the Naive Bayes code.
But finally we don't use this method.
"""

if __name__ == "__main__":
    loader_pos = Loader(FILE_LOC_POS)
    loader_neg = Loader(FILE_LOC_NEG)
    pos_dict = copy.deepcopy(loader_pos.dict)
    neg_dict = copy.deepcopy(loader_neg.dict)
    all_dict = dict(Counter(pos_dict) + Counter(neg_dict))
    number_pos = len(loader_pos.sentences)
    number_neg = len(loader_neg.sentences)
    print(number_pos)
    print(number_neg)

    correct = 0
    for test_sentence in loader_neg.sentences:
        # print(test_sentence)

        s = set()
        sum_pos = math.log(1.0 * number_pos / (number_pos + number_neg))
        C = loader_pos.total_words + len(all_dict)
        for word in test_sentence:
            if not word in s:
                try:
                    if pos_dict[word] > 0:
                        sum_pos = sum_pos + \
                            math.log(
                                (1.0 * pos_dict[word] + 1) / C)
                except KeyError:
                    sum_pos = sum_pos + math.log(1.0 * 1 / C)
        # print(sum)

        s = set()
        sum_neg = math.log(1.0 * number_neg / (number_pos + number_neg))
        C = loader_neg.total_words + len(all_dict)
        for word in test_sentence:
            if not word in s:
                try:
                    if neg_dict[word] > 0:
                        sum_neg = sum_neg + \
                            math.log(
                                (1.0 * neg_dict[word] + 1) / C)
                except KeyError:
                    sum_neg = sum_neg + math.log(1.0 * 1 / C)
        # print(sum)

        # print(sum_pos, sum_neg)

        if (sum_pos < sum_neg):
            correct = correct + 1
    print(correct / number_neg)

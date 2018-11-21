import gensim.downloader as api
import numpy as np
import pickle
import spacy
import torch
import torch.nn

from collections import OrderedDict

from modules.utilities import my_dependencies
from modules.data_iterators import STSDataIterator

# STS
# train_freqs = {'ROOT': 11420, 'nsubj': 10694, 'nummod': 2328, 'amod': 7866, 'quantmod': 304, 'prep': 12730, 'pobj': 12462, 'det': 12286, 'aux': 4698, 'punct': 12246, 'dobj': 6402, 'cc': 1994, 'conj': 2408, 'neg': 426, 'compound': 10790, 'acl': 1458, 'pcomp': 330, 'advmod': 2102, 'npadvmod': 982, 'nmod': 744, 'appos': 692, 'xcomp': 688, 'advcl': 934, 'poss': 1576, 'case': 940, 'nsubjpass': 660, 'auxpass': 726, 'ccomp': 1314, 'prt': 512, 'mark': 746, 'attr': 542, 'csubj': 36, 'relcl': 542, 'agent': 198, 'acomp': 398, 'dative': 78, 'oprd': 94, '': 64, 'intj': 102, 'parataxis': 18, 'meta': 6, 'dep': 74, 'preconj': 18, 'expl': 64, 'predet': 16}
# dev_freqs = {'ROOT': 2938, 'nsubj': 3236, 'dobj': 1824, 'det': 4132, 'acl': 350, 'advmod': 1030, 'nsubjpass': 180, 'auxpass': 196, 'prt': 172, 'punct': 4110, 'cc': 684, 'conj': 782, 'poss': 484, 'compound': 2118, 'case': 184, 'amod': 2186, 'nmod': 146, 'prep': 3702, 'pobj': 3584, 'dative': 18, 'aux': 1626, 'acomp': 268, 'neg': 194, 'advcl': 372, 'mark': 332, 'relcl': 240, 'attr': 286, 'ccomp': 436, 'xcomp': 306, 'pcomp': 142, 'npadvmod': 228, 'nummod': 574, 'intj': 62, 'appos': 186, 'csubj': 22, 'quantmod': 96, 'expl': 88, 'predet': 8, 'agent': 60, 'parataxis': 20, 'oprd': 28, 'dep': 14, '': 6, 'preconj': 4, 'meta': 4}
# test_freqs = {'ROOT': 2236, 'nsubj': 2402, 'neg': 178, 'attr': 192, 'det': 3272, 'amod': 1270, 'punct': 3116, 'aux': 1406, 'dobj': 1398, 'prep': 2316, 'pobj': 2222, 'poss': 372, 'compound': 1136, 'case': 108, 'appos': 112, 'nummod': 516, 'cc': 504, 'conj': 542, 'acl': 276, 'advmod': 586, 'acomp': 104, 'xcomp': 240, 'ccomp': 330, 'nsubjpass': 136, 'auxpass': 150, 'agent': 36, 'relcl': 128, 'advcl': 182, 'mark': 124, 'npadvmod': 188, 'quantmod': 60, 'nmod': 108, 'pcomp': 106, 'prt': 110, 'intj': 42, 'parataxis': 2, 'expl': 82, 'oprd': 16, 'dative': 20, '': 6, 'csubj': 2}

# SICK
train_freqs = {'ROOT': 9000, 'nsubj': 8538, 'det': 18771, 'prep': 8823, 'pobj': 9211, 'aux': 8809, 'compound': 1703, 'cc': 2054, 'conj': 2127, 'dobj': 5059, 'poss': 463, 'advmod': 968, 'amod': 4709, 'nummod': 1047, 'neg': 416, 'relcl': 399, 'acomp': 261, 'nsubjpass': 549, 'auxpass': 584, 'agent': 452, 'prt': 417, 'attr': 642, 'expl': 562, 'acl': 804, 'npadvmod': 35, 'punct': 461, 'case': 55, 'ccomp': 42, 'quantmod': 10, 'advcl': 68, 'pcomp': 6, 'xcomp': 47, 'nmod': 30, 'dative': 13, 'mark': 7, 'appos': 18, 'oprd': 5, 'dep': 6, 'intj': 5, 'csubj': 2}
test_freqs = {'ROOT': 9854, 'nsubj': 9411, 'det': 20335, 'aux': 9682, 'dobj': 5526, 'prep': 9672, 'pobj': 10029, 'nummod': 1214, 'nsubjpass': 571, 'amod': 5353, 'auxpass': 623, 'agent': 450, 'expl': 580, 'attr': 630, 'compound': 1753, 'acl': 837, 'relcl': 432, 'cc': 2297, 'conj': 2388, 'advmod': 1068, 'poss': 495, 'prt': 496, 'acomp': 253, 'punct': 515, 'dative': 20, 'advcl': 81, 'appos': 27, 'neg': 447, 'ccomp': 42, 'xcomp': 54, 'case': 48, 'nmod': 25, 'npadvmod': 36, 'oprd': 7, 'quantmod': 9, 'mark': 9, 'dep': 10, 'csubj': 9, 'pcomp': 12, 'intj': 5, '': 2}

nlp = spacy.load('en')

sts_data = './data/stsbenchmark/'
sts_train_di = STSDataIterator(sts_data+'train_data.pkl', 1)
sts_dev_di = STSDataIterator(sts_data+'dev_data.pkl', 1)
sts_test_di = STSDataIterator(sts_data+'test_data.pkl', 1)

sick_data = './data/sick/'
sick_train_di = STSDataIterator(sick_data+'train.pkl', 1)
sick_test_di = STSDataIterator(sick_data+'test.pkl', 1)


def add_dep_freq(dep, dep_freqs):
    if dep in dep_freqs:
        dep_freqs[dep] += 1
    else:
        dep_freqs[dep] = 1

def add_dep_freqs(node, dep_freqs):
    add_dep_freq(node.dep_, dep_freqs)
    for c in node.children:
        add_dep_freqs(c, dep_freqs)

def get_dep_freqs(di):
    dep_freqs = {}
    for i, example in enumerate(iter(di)):
        s1, s2 = str(example[0][0]), str(example[0][1])
        r1 = list(nlp(s1).sents)[0].root
        r2 = list(nlp(s2).sents)[0].root
        add_dep_freqs(r1, dep_freqs)
        add_dep_freqs(r2, dep_freqs)

    return dep_freqs

def create_tree_dataset(data):
    word_embeddings = api.load('glove-wiki-gigaword-50')
    # now we will create an object that stores sentences as EmbeddingTrees rather than strings,
    # but otherwise will be identical to the OG dataset.

def recur(node):
    if node.dep_ == "dep":
        print(node.sent)
    for c in node.children:
        recur(c)

def print_ordered_freqs_and_ratios():
    otf = OrderedDict()
    for k in sorted(train_freqs.keys()):
        otf[k] = train_freqs[k]
    print(otf)
    # otf = OrderedDict()
    # for k in sorted(dev_freqs.keys()):
    #     otf[k] = dev_freqs[k]
    # print(otf)
    otf = OrderedDict()
    for k in sorted(test_freqs.keys()):
        otf[k] = test_freqs[k]
    print(otf)

    train_freq_ratios = OrderedDict()
    root_freq = train_freqs['ROOT']
    for k in sorted(train_freqs.keys()):
        train_freq_ratios[k] = round(float(train_freqs[k]) / root_freq, 3)
    print(train_freq_ratios)
    # dev_freq_ratios = OrderedDict()
    # root_freq = dev_freqs['ROOT']
    # for k in sorted(dev_freqs.keys()):
    #     dev_freq_ratios[k] = round(float(dev_freqs[k]) / root_freq, 3)
    # print(dev_freq_ratios)
    test_freq_ratios = OrderedDict()
    root_freq = test_freqs['ROOT']
    for k in sorted(test_freqs.keys()):
        test_freq_ratios[k] = round(float(test_freqs[k]) / root_freq, 3)
    print(test_freq_ratios)

if __name__ == "__main__":
    # train_freqs = get_dep_freqs(sick_train_di)
    # dev_freqs = get_dep_freqs(dev_di)
    # test_freqs = get_dep_freqs(sick_test_di)

    # print_ordered_freqs_and_ratios()

    # for i, example in enumerate(iter(train_di)):
    #     s1, s2 = str(example[0][0]), str(example[0][1])
    #     r1 = list(nlp(s1).sents)[0].root
    #     r2 = list(nlp(s2).sents)[0].root
    #     recur(r1)
    #     recur(r2)


    # TO WORK OUT WHICH DEPENDENCIES ARE IN SICK

    # ignored_dependencies = ['expl', 'punct', 'root', 'intj', 'meta', 'oprd', 'predet', 'parataxis', 'dative', 'agent', 'quantmod', 'det']

    # train_deps = set()
    # for k, v in train_freqs.items():
    #     if k not in train_deps:
    #         train_deps.add(k)
    # test_deps = set()
    # for k, v in test_freqs.items():
    #     if k not in test_deps:
    #         test_deps.add(k)

    # mine = set()
    # for k, v in my_dependencies.items():
    #     if k not in mine:
    #         mine.add(k)
        # if k not in train_deps:
        #     print("dependency {0} not seen in SICK training data".format(k))
        # if k not in train_deps:
        #     print("dependency {0} not seen in SICK testing data".format(k))
    
    # for k in train_freqs.keys():
    #     if k not in mine and k not in ignored_dependencies:
    #         print(k)
    # print('---')
    # for k in test_freqs.keys():
    #     if k not in mine and k not in ignored_dependencies:
    #         print(k)
        

universal = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']

universal_tags = {
    'ADV': 2, 'NOUN': 8, 'ADP': 1, 'PUNCT': 13, 'SCONJ': 14,
    'PROPN': 12, 'SPACE': 18, 'DET': 6, 'SYM': 15, 'INTJ': 7,
    'PART': 10, 'PRON': 11, 'NUM': 9, 'X': 17, 'CCONJ': 5, 'AUX': 3,
    'CONJ': 4, 'ADJ': 0, 'VERB': 16
}

nouns = ['NN',  'NNS']
proper_nouns = ['NNP', 'NNPS']
verbs = ['BES', 'HVS', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# uni = {}
# for i, x in enumerate(universal):
#     uni[x] = i
# print(uni)

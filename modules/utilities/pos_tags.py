universal = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']

universal_tags = {
    'ADV': 2, 'NOUN': 8, 'ADP': 1, 'PUNCT': 13, 'SCONJ': 14,
    'PROPN': 12, 'SPACE': 18, 'DET': 6, 'SYM': 15, 'INTJ': 7,
    'PART': 10, 'PRON': 11, 'NUM': 9, 'X': 17, 'CCONJ': 5, 'AUX': 3,
    'CONJ': 4, 'ADJ': 0, 'VERB': 16
}

english = [
    '-LRB-', '-RRB', ',', ':', '.', "'", "\"", "#", "$", "ADD", "FW", "GW",
    "AFX", "BES", "CC", "CD", "DT", "EX", "HVS", "HYPH", "IN", "JJ", "JJR",
    "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "POS", "PRP", "PDT",
    "RB", "RBR", "RBS", "RP", "_SP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
    "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", "XX"
]
punctuation = ['-LRB-', '-RRB', ',', ':', '.', "'", "\"", "HYPH", "LS", "NFP"]
symbols = ["#", "$", "SYM"]
xs = ['ADD', 'FW', 'GW', "XX"]
nums = ["CD"]
adjs = ["JJ", "JJR", "JJS"]
advbs = ["RB", "RBR", "RBS"]
nouns = ["NN", "NNP", "NNPS", "NNS"]

english_tags = {
    '-LRB-': 0, '-RRB': 0, ',': 0, ':': 0, '.': 0, "'": 0, '"': 0, 'HYPH': 0,
    'LS': 0, 'NFP': 0, '#': 1, '$': 1, 'ADD': 2, 'FW': 2, 'GW': 2, 'CD': 3, 'JJ': 4,
    'JJR': 4, 'JJS': 4, 'RB': 5, 'RBR': 5, 'RBS': 5, 'NN': 6, 'NNP': 6, 'NNPS': 6,
    'NNS': 6, 'AFX': 7, 'BES': 8, 'CC': 9, 'DT': 10, 'EX': 11, 'HVS': 12, 'IN': 13,
    'MD': 14, 'POS': 15, 'PRP': 16, 'PDT': 17, 'RP': 18, '_SP': 19, 'SYM': 1, 'TO': 20,
    'UH': 21, 'VB': 22, 'VBD': 23, 'VBG': 24, 'VBN': 25, 'VBP': 26, 'VBZ': 27,
    'WDT': 28, 'WP': 29, 'WRB': 30, 'XX': 2
}

def get_english_tags():
    count = 0
    tags = {}
    for t in punctuation:
        tags[t] = count
    count += 1
    for t in symbols:
        tags[t] = count
    count += 1
    for t in xs:
        tags[t] = count
    count += 1
    for t in nums:
        tags[t] = count
    count += 1
    for t in adjs:
        tags[t] = count
    count += 1
    for t in advbs:
        tags[t] = count
    count += 1
    for t in nouns:
        tags[t] = count
    count += 1
    for t in english:
        if t not in tags.keys():
            tags[t] = count
            count += 1
    
    return tags
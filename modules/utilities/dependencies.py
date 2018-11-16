# universal dependencies
universal_dependencies = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp',
                'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'dislocated',
                'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj',
                'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root',
                'vocative', 'xcomp'] # 37 dependencies

# ignored dependencies
ignored_dependencies = ['clf', 'discourse', 'dislocated', 'expl', 'punct', 'reparandum', 'root']

# multi-word dependencies
multi_word_dependencies = ['compound', 'fixed', 'flat', 'goeswith']

# modify a verb, adjective, or adverb
adverbial_dependencies = ['advcl', 'advmod', 'obl']

# modify a noun or noun-phrase
noun_phrase_dependencies = ['acl', 'amod', 'nmod', 'nummod']

# subject dependencies
subj_dependencies = ['csubj', 'nsubj']

# mapping of dependencies to the indices of parameters they'll use
my_dependencies = {
    'advcl': 0, 'advmod': 0, 'obl': 0, 'acl': 1, 'amod': 1, 'nmod': 1, 'nummod': 1,
    'compound': 2, 'fixed': 2, 'flat': 2, 'goeswith': 2, 'csubj': 3, 'nsubj': 3,
    'appos': 4, 'aux': 5, 'case': 6, 'cc': 7, 'ccomp': 8, 'conj': 9, 'cop': 10,
    'dep': 11, 'det': 12, 'iobj': 13, 'list': 14, 'mark': 15, 'obj': 16, 'orphan': 17,
    'parataxis': 18, 'vocative': 19, 'xcomp': 20
}

def get_my_dependencies():
    """
    How to get these dependencies. The idea is that we will use the same params
        for dependencies that are functionally equivalent given a universal representation
        for a sequence of text (which is what the DependencyNetwork is aiming to produce)
    """
    count = 0
    ds = {}
    for dep in adverbial_dependencies:
        ds[dep] = count
    count += 1
    for dep in noun_phrase_dependencies:
        ds[dep] = count
    count += 1
    for dep in multi_word_dependencies:
        ds[dep] = count
    count += 1
    for dep in subj_dependencies:
        ds[dep] = count
    count += 1
    for dep in dependencies:
        if dep not in ds.keys() and dep not in ignored_dependencies:
            ds[dep] = count
            count += 1
    
    my_dependencies = ds
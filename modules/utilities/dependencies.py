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

# NEW ONES, FOR ENGLISH DEPENDENCIES
english_dependencies = ['acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux',
                'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative',
                'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'npmod', 'nsubj',
                'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet',
                'prep', 'prt', 'punct', 'quantmod', 'relcl', 'root', 'xcomp'] # 45 dependencies

# added dependencies to the ignored list if:
    # 1) <3500 occurrences on 10 epochs of training on STS-benchmark data - arbitrary, but they're just v rare
    # 2) punct/root can explain why I don't want to consider these.
ignored_dependencies = ['expl', 'punct', 'root', 'intj', 'meta', 'oprd', 'predet', 'parataxis', 'dative', 'agent', 'quantmod', 'det']
subj_dependencies = ['csubj', 'nsubj']
subj_pass_dependencies = ['csubjpass', 'nsubjpass']
prepositional_dependencies = ['pobj', 'pcomp']
adverbial_dependencies = ['advmod', 'npadvmod', 'advcl']


# mapping of dependencies to the indices of parameters they'll use
my_dependencies = {
    'advcl': 0, 'advmod': 0, 'obl': 0, 'acl': 1, 'amod': 1, 'nmod': 1, 'nummod': 1,
    'compound': 2, 'fixed': 2, 'flat': 2, 'goeswith': 2, 'csubj': 3, 'nsubj': 3,
    'appos': 4, 'aux': 5, 'case': 6, 'cc': 7, 'ccomp': 8, 'conj': 9, 'cop': 10,
    'dep': 11, 'det': 12, 'iobj': 13, 'list': 14, 'mark': 15, 'obj': 16, 'orphan': 17,
    'parataxis': 18, 'vocative': 19, 'xcomp': 20
}


# The new mapping coming from English-specific dependencies
my_dependencies = {
    'advmod': 0, 'npadvmod': 0, 'advcl': 0, 'pobj': 1, 'pcomp': 1,
    'csubj': 2, 'nsubj': 2, 'csubjpass': 3, 'nsubjpass': 3, 'acl': 4,
    'acomp': 5, 'amod': 6, 'appos': 7, 'attr': 8, 'aux': 9, 'auxpass': 10,
    'case': 11, 'cc': 12, 'ccomp': 13, 'compound': 14, 'conj': 15, 'dep': 16,
    'dobj': 17, 'mark': 18, 'neg': 19, 'npmod': 20, 'nummod': 21, 'poss': 22,
    'preconj': 23, 'prep': 24, 'prt': 25, 'relcl': 26, 'xcomp': 27
}
# NB: csubjpass, npmod, preconj all not seen in SICK dataset at all - everything else is.

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
    for dep in prepositional_dependencies:
        ds[dep] = count
    count += 1
    for dep in subj_dependencies:
        ds[dep] = count
    count += 1
    for dep in subj_pass_dependencies:
        ds[dep] = count
    count += 1
    for dep in english_dependencies:
        if dep not in ds.keys() and dep not in ignored_dependencies:
            ds[dep] = count
            count += 1
    
    return ds

if __name__ == "__main__":
    print(get_my_dependencies())
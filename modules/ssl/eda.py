"""
To implement easy data augmentation techniques from EDA (Wei & Zou 2019).
"""
from nltk.corpus import wordnet

from modules.utilities.imports import *

__all__ = ['eda', 'eda_corpus']


STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']
CHARS = ' qwertyuiopasdfghjklzxcvbnm'


def get_only_chars(sentence):
    sentence = sentence.replace("’", "")
    sentence = sentence.replace("'", "")
    sentence = sentence.replace("-", " ")
    sentence = sentence.replace("\t", " ")
    sentence = sentence.replace("\n", " ")
    sentence = sentence.lower()

    clean_chars = [char if char in CHARS else ' ' for char in sentence]
    clean_sentence = ''.join(clean_chars)
    clean_sentence = re.sub(' +', ' ', clean_sentence) # delete extra spaces
    if clean_sentence[0] == ' ':
        clean_sentence = clean_sentence[1:]
    
    return clean_sentence


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in CHARS])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in STOP_WORDS]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    
    return new_words


def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        return [words[random.randint(0, len(words)-1)]]
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10: return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def eda(words, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, n_aug=9):
    sentence = ' '.join(words)
    sentence = get_only_chars(sentence)
    words = [word for word in sentence.split(' ') if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(n_aug/4)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, alpha_rd)
        augmented_sentences.append(' '.join(a_words))
    
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    if n_aug >= 1:
        augmented_sentences = augmented_sentences[:n_aug]
    else:
        keep_prob = n_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
    
    augmented_sentences.append(sentence)
    augmented_sentences = [sentence.split(' ') for sentence in augmented_sentences]

    return augmented_sentences


def eda_synonyms(words, alpha_sr=0.1, n_aug=9):
    sentence = ' '.join(words)
    sentence = get_only_chars(sentence)
    words = [word for word in sentence.split(' ') if word is not '']
    num_words = len(words)

    augmented_sentences = []
    n_sr = max(1, int(alpha_sr*num_words))
    for _ in range(n_aug):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    if n_aug >= 1:
        augmented_sentences = augmented_sentences[:n_aug]
    else:
        keep_prob = n_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(sentence)
    augmented_sentences = [sentence.split(' ') for sentence in augmented_sentences]

    return augmented_sentences


def eda_corpus(x_l, y_l, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, n_aug=3):
    new_x_l, new_y_l = [], []
    for x, y in zip(x_l, y_l):
        # aug_sentences = eda(x, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, n_aug=n_aug)
        aug_sentences = eda_synonyms(x, alpha_sr=alpha_sr, n_aug=n_aug)
        new_x_l += aug_sentences
        new_y_l += [y for i in range(len(aug_sentences))]

    return new_x_l, new_y_l

from flair.data import Sentence
from flair.embeddings import BertEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings
from sklearn.model_selection import ParameterGrid
from torchtext import data
from torchtext.data.example import Example

from modules.data_iterators import get_ic_data_iterators
from modules.models import create_encoder
from modules.model_wrappers import IntentWrapper
from modules.train import do_basic_train_and_classify, self_feed
from modules.utilities import repeat_trainer, traina

from modules.utilities.imports import *
from modules.utilities.imports_torch import *

from .lp import LabelProp
from .knn import knn_classify

__all__ = ['repeat_augment_and_train']


def repeat_augment_and_train(dir_to_save, data_source, aug_algo, encoder_model, sim_measure, datasets, text_field, label_field, frac, num_classes, classifier_params, k=10):
    """
    Runs k trials of augmentation & repeat-classification for a given fraction of labeled training data.
    Args:
        dir_to_save (str): directory to save models created/loaded during this process
        aug_algo (str): which augmentation algorithm to use
        encoder_model (str): encoder model to use for augmentation (w similarity measure between these encodings)
        sim_measure (str): which similarity measure to use
        datasets (list(Dataset)): train/val/test torchtext datasets
        text_field (Field): torchtext field for sentences
        label_field (LabelField): torchtext LabelField for class labels
        frac (float): Fraction of labeled training data to use
        classifier_params (dict): params for intent classifier to use on augmented data.
        k (int): Number of times to repeat augmentation-classifier training process
    Returns:
        8 statistical measures of the results of these trials
    """
    if data_source in ('chatbot', 'webapps', 'askubuntu'):
        k = 40

    ###########################################################################################################################################################
    ###########################################################################################################################################################
    # k = 5 # TO TRY STS^-1 distance measure
    ###########################################################################################################################################################
    ###########################################################################################################################################################

    train_ds, val_ds, test_ds = datasets
    class_accs, aug_accs, aug_fracs = [], [], []
    ps, rs, fs = [], [], []

    for i in tqdm(range(k), total=k):
        examples = train_ds.examples
        np.random.shuffle(examples)
        cutoff = int(frac*len(examples))

        if data_source in ('askubuntu', 'webapps'):
            if frac == 0: # SPECIAL CASE TO TEST 1 from every class
                classes_seen = {i: 0 for i in range(num_classes)}
                labeled_examples, unlabeled_examples = [], []
                for eg in examples:
                    if classes_seen[eg.y] == 0:
                        labeled_examples.append(eg)
                        classes_seen[eg.y] += 1
                    else:
                        unlabeled_examples.append(eg)
            else: # ensure at least one labeled example from each class
                while True:
                    labeled_examples = examples[:cutoff]
                    unlabeled_examples = examples[cutoff:]
                    if len(set([eg.y for eg in labeled_examples])) == num_classes:
                        break
                    np.random.shuffle(examples)
        else:
            labeled_examples = examples[:cutoff]
            unlabeled_examples = examples[cutoff:]

        if aug_algo == "none" or frac == 1:
            augmented_train_examples = labeled_examples
            aug_acc = 1; frac_used = 0
        elif aug_algo == "self_feed":
            augmented_train_examples, aug_acc, frac_used = self_feed(data_source, dir_to_save, labeled_examples, unlabeled_examples, val_ds, test_ds, text_field, label_field, classifier_params)
        else:
            augmented_train_examples, aug_acc, frac_used = augment(data_source, aug_algo, encoder_model, sim_measure, labeled_examples, unlabeled_examples, train_ds, text_field, label_field, num_classes)
        
        aug_accs.append(aug_acc); aug_fracs.append(frac_used)
        new_train_ds = data.Dataset(augmented_train_examples, {'x': text_field, 'y': label_field})
        new_datasets = (new_train_ds, val_ds, test_ds)

        if data_source in ('chatbot', 'webapps', 'askubuntu'):
            acc, p, r, f = do_basic_train_and_classify(new_train_ds, test_ds, classifier_params)
        else:
            acc, p, r, f = train_ic(dir_to_save, text_field, label_field, new_datasets, classifier_params, return_statistics=True)
        class_accs.append(acc); ps.append(p); rs.append(r); fs.append(f)

    print(f"FRAC '{frac}' Results Below:")
    print(f'classification acc --> mean: {np.mean(class_accs)}, std: {np.std(class_accs)}')
    print(f'augmentation acc --> mean: {np.mean(aug_accs)}; std: {np.std(aug_accs)}\t (average frac used: {np.mean(aug_fracs)})')
    print(f'p/r/f1 --> precision mean: {np.mean(ps)}; recall mean: {np.mean(rs)}; f1 mean: {np.mean(fs)}')
    print(f'p/r/f1 --> precision std: {np.std(ps)}; recall std: {np.std(rs)}; f1 std: {np.std(fs)}')

    class_acc_mean, class_acc_std = np.mean(class_accs), np.std(class_accs)
    aug_acc_mean, aug_acc_std, aug_frac_mean = np.mean(aug_accs), np.std(aug_accs), np.mean(aug_fracs)
    p_mean, r_mean, f_mean = np.mean(ps), np.mean(rs), np.mean(fs)
    p_std, r_std, f_std = np.std(ps), np.std(rs), np.std(fs)

    return class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, aug_frac_mean, p_mean, p_std, r_mean, r_std, f_mean, f_std


def augment(data_source, aug_algo, encoder_model, sim_measure, labeled_examples, unlabeled_examples, train_ds, text_field, label_field, num_classes):
    if encoder_model.startswith('pretrained'):
        embedding_type = encoder_model.split('_')[1]
        res = encode_data_with_pretrained(data_source, train_ds, text_field, embedding_type, labeled_examples, unlabeled_examples)
    elif encoder_model.startswith('sts'):
        res = encode_data_with_pretrained(data_source, train_ds, text_field, encoder_model, labeled_examples, unlabeled_examples)
    xs_l, ys_l, xs_u, ys_u, xs_u_unencoded = res

    if sim_measure != "sts":
        xs_l = np.array([x / np.linalg.norm(x) for x in xs_l])
        xs_u = np.array([x / np.linalg.norm(x) for x in xs_u])
    
    if aug_algo.startswith("knn"):
        algo_version = aug_algo.split('_')[1]
        if algo_version == 'base':
            classifications, indices = knn_classify(xs_l, ys_l, xs_u, n=1, weights='uniform')
            frac_used = 1
        elif algo_version == 'threshold':
            classifications, indices = knn_classify(xs_l, ys_l, xs_u, n=2, threshold=0.99)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
    elif aug_algo.startswith("lp"):
        algo_version = aug_algo.split('_')[1]
        lp = LabelProp(xs_l, ys_l, xs_u, num_classes, sim_measure=sim_measure, source=encoder_model.split('_')[1])
        if algo_version == 'base':
            lp.propagate()
            classifications, indices = lp.classify(threshold=False)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
        elif algo_version == "threshold":
            lp.propagate()
            classifications, indices = lp.classify(threshold=True)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
        elif algo_version == "recursive":
            classifications, indices = lp.recursive(xs_l, ys_l, xs_u, ys_u)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
        elif algo_version == "p1nn":
            classifications, indices = lp.p1nn(xs_l, ys_l, xs_u)
            frac_used = 1
    
    num_correct = np.sum(classifications == ys_u)
    aug_acc = 0 if len(classifications) == 0 else float(num_correct/len(classifications))
        
    new_labeled_data = [{'x': x, 'y': classifications[i]} for i, x in enumerate(xs_u_unencoded)]
    example_fields = {'x': ('x', text_field), 'y': ('y', label_field)}
    new_examples = [Example.fromdict(x, example_fields) for x in new_labeled_data]

    return labeled_examples + new_examples, aug_acc, frac_used


def encode_data_with_pretrained(data_source, train_ds, text_field, embedding_type, examples_l, examples_u):
    data_source_embeddings_path = f'./data/ic/{data_source}/{embedding_type}_embeddings.pkl'
    embeddings_file = Path(data_source_embeddings_path)
    
    if not embeddings_file.is_file():
        create_pretrained_embeddings(train_ds, text_field, data_source, embedding_type)
        embeddings_file = Path(data_source_embeddings_path)

    embeddings = pickle.load(embeddings_file.open('rb'))
    xs_l = np.array([embeddings[' '.join(eg.x)] for eg in examples_l])
    xs_u = np.array([embeddings[' '.join(eg.x)] for eg in examples_u])

    ys_l = np.array([eg.y for eg in examples_l])
    ys_u = np.array([eg.y for eg in examples_u])
    xs_u_unencoded = [eg.x for eg in examples_u]

    return xs_l, ys_l, xs_u, ys_u, xs_u_unencoded


def create_pretrained_embeddings(train_ds, text_field, data_source, embedding_type):
    if embedding_type == "glove":
        encoder = create_encoder(text_field.vocab, 300, "pool_max", *['max'])
        encoder.eval()
        sents = [torch.tensor([[text_field.vocab.stoi[t] for t in eg.x]]) for eg in train_ds.examples]
        embeddings = {}
        for eg, idxed in zip(train_ds.examples, sents):
            sent = ' '.join(eg.x)
            emb = encoder(idxed.reshape(-1, 1)).detach().squeeze(0).numpy()
            embeddings[sent] = emb
    
    elif embedding_type.startswith("sts"):
        source = embedding_type.split('_')[1]
        enc_params = pickle.load(Path(f'./data/sts/{source}/pretrained/params.pkl').open('rb'))['encoder']
        emb_dim, hid_dim = enc_params['emb_dim'], enc_params['hid_dim']
        num_layers, output_type = enc_params['num_layers'], enc_params['output_type']
        bidir, fine_tune = enc_params['bidir'], enc_params['fine_tune']
        vocab = pickle.load(Path(f'./data/sts/{source}/pretrained/vocab.pkl').open('rb'))
        bidir = True # this is set incorrectly in stsbenchmark params
        encoder = create_encoder(vocab, emb_dim, "lstm", *[hid_dim, num_layers, bidir, fine_tune, output_type])
        encoder.load_state_dict(torch.load(f'./data/sts/{source}/pretrained/encoder.pt', map_location=lambda storage, loc: storage))
        encoder.eval()
        
        sents = [torch.tensor([[vocab.stoi[t] for t in eg.x]]).reshape(-1,1) for eg in train_ds.examples]
        embeddings = {}
        for eg, idxed in zip(train_ds.examples, sents):
            sent = ' '.join(eg.x)
            emb = encoder(idxed).detach().squeeze(0).numpy()
            embeddings[sent] = emb

    elif embedding_type == "infersent":
        from pretrained_models.infersent.models import InferSent
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        model = InferSent(params_model)
        model.load_state_dict(torch.load('/Users/michaelball/Desktop/Thesis/repo/pretrained_models/infersent/infersent1.pkl'))
        model.set_w2v_path('/Users/michaelball/Desktop/Thesis/repo/data/glove.840B.300d.txt')
        sentences = [' '.join(eg.x) for eg in train_ds.examples]
        model.build_vocab(sentences, tokenize=False)
        emb = model.encode(sentences, bsize=128, tokenize=False, verbose=False)
        embeddings = {s:e for s,e in zip(sentences, emb)}
    
    elif embedding_type == "bert" or embedding_type == "elmo":
        encoder = {"bert": BertEmbeddings(), "elmo": ELMoEmbeddings()}[embedding_type]
        sents = [Sentence(' '.join(eg.x)) for eg in train_ds.examples]
        encoder.embed(sents)
        embs = np.array([torch.max(torch.stack([t.embedding for t in S]), 0)[0].detach().numpy() for S in sents])
        embeddings = {' '.join(eg.x): emb for eg, emb in zip(train_ds.examples, embs)}

    pickle.dump(embeddings, Path(f'/Users/michaelball/Desktop/Thesis/repo/data/ic/{data_source}/{embedding_type}_embeddings.pkl').open('wb'))


def train_ic(dir_to_save, text_field, label_field, datasets, classifier_params, return_statistics=False):
    """
    To simplify training (connection to generic training utility with appropriate parameters).
    # ULTIMATELY, classifier_params should just be able to be passed to this train method along with other
    # stuff and THAT method is what sorts it all out... but this is fine for now.
    """
    ps = classifier_params
    wrapper = traina(ps['model_name'], ps['encoder_model'], get_ic_data_iterators, IntentWrapper, dir_to_save,
                    nn.CrossEntropyLoss(), datasets, text_field, ps['bs'], ps['encoder_args'],
                    ps['layers'], ps['drops'], ps['lr'], frac=1, verbose=False)
    if return_statistics:
        acc = wrapper.test_accuracy(wrapper.test_di)
        p, r, f, _ = wrapper.test_precision_recall_f1()
        return acc, np.mean(p), np.mean(r), np.mean(f)
    return wrapper


def repeat_ic(dir_to_save, text_field, label_field, datasets, classifier_params, k=10):
    """
    Repeat intent classification training process
    """
    loss_func = nn.CrossEntropyLoss()
    ps = classifier_params
    mean, std, avg_p, avg_r, avg_f = repeat_trainer(ps['model_name'], ps['encoder_model'], get_ic_data_iterators, IntentWrapper, dir_to_save, 
                                    loss_func, datasets, text_field, label_field, ps['bs'], ps['encoder_args'],
                                    ps['layers'], ps['drops'], ps['lr'], frac=1, k=k, verbose=True)
    return mean, std, avg_p, avg_r, avg_f

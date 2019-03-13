import argparse

from torchtext import data, vocab
from torchtext.data.example import Example

from modules.data_iterators import *
from modules.data_readers import *
from modules.models import *
from modules.model_wrappers import *
from modules.ssl import *
from modules.utilities import *

from modules.utilities.imports import *
from modules.utilities.imports_torch import *


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--subtask', type=str, default='none', help='sub-task within whichever specified task')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--model_name', type=str, default='999', help='name of model to train')
parser.add_argument('--encoder_model', type=str, default='pos_tree', help='encoder model for primary task')
parser.add_argument('--frac', type=float, default=1, help='fraction of training data to use')
parser.add_argument('--predictor_model', type=str, default='mlp', help='mlp / cosine_sim')
parser.add_argument('--aug_algo', type=str, default='none', help='none|knn_all|my_lp|self_feed|')
parser.add_argument('--sim_measure', type=str, default='cosine', help='cosine|sts')
args = parser.parse_args()


def get_results(algorithm, data_source, classifier, encoder=None, similarity_measure=None):
    """
    Get trial results for a given experiment setup from results file.
    Args:
        algorithm (str): algo used for learning
        data_source (str): for which dataset
        encoder (str): encoder used (if SSL) in trials
        similarity_measure (str): similarity measure used (if SSL)
        classifier (str): type of classifier used in trials`
    Returns:
        Dictionary of statistical results (accuracy means, stds, f1s, etc).
    """
    from results_chat import results_chat
    from results_askubuntu import results_askubuntu
    from results_chatbot import results_chatbot
    from results_webapps import results_webapps
    r = {'chat': results_chat, 'askubuntu': results_askubuntu, 'chatbot': results_chatbot, 'webapps': results_webapps}

    if algorithm is 'supervised' or algorithm is 'self_feed':
        results_name = f'{algorithm}__{classifier}'
    else:
        results_name = f'{algorithm}__{encoder}__{similarity_measure}__{classifier}'
    
    return r[data_source][results_name]


if __name__ == "__main__":
    
    # MODELS TO TRY for STS/NLI:
    # Idea: use pre-trained Infersent and then just put a big classifier over this? Because Infersent is trained on SNLI, => should express similarity better.
    # easiest way is to do the same thing, encode all sentences in our IC datasets and just access these values?
    # Multi-Task Learning framework (two STS datasets, two NLI datasets) - => two heads with one common encoder (a big, BiLSTM ? initialized with GloVe).

    # INTENT CLASSIFICATION WITH SEMI_SUPERVISED LEARNING
    if args.task.startswith("propagater"):
        # params
        t = args.task.split("_")
        task = t[0]; data_source = t[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        EMB_DIM = 300

        # we are getting data augmentation w 99% accuracy with lp_recursive... but NB isn't getting good acc or f1 with this => maybe need a different algorithm for some reason. -- this is for ask_ubuntu
        if data_source in ('chatbot', 'webapps', 'askubuntu'):
            classifier = 'sgd' if data_source is 'webapps' else 'nb'
            class_args = {
                'askubuntu': {'alpha': 1},
                'chatbot': {'alpha': 0.1},
                'webapps': {'loss': 'hinge', 'penalty': 'l2', 'alpha': 0.01, 'learning_rate': 'optimal'},
            }[data_source]
            classifier_params = {
                'vectoriser': 'count',
                'transformer': 'tfidf',
                'classifier': classifier,
                'vect_args': {'analyzer': 'char', 'binary': False, 'max_df': 0.5, 'ngram_range': (2,3)},
                'trans_args': {'norm': 'l2', 'use_idf': True},
                'class_args': class_args
            }
        else:
            classifier_params = {
                'model_name': 'test',
                'encoder_model': 'pool_max',
                'encoder_args': ['max'],
                'emb_dim': EMB_DIM,
                'layers': [EMB_DIM, 100, C],
                'drops': [0, 0],
                'bs': 64,
                'lr': 6e-4
            }

        # create datasets/vocab
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        val = False if data_source in ('webapps', 'chatbot', 'askubuntu') else True
        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL, val=val).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        # run augmentation trials
        aug_algo = args.aug_algo
        dir_to_save = f'{args.saved_models}/ic/{data_source}'
        class_acc_means, class_acc_stds = [], []
        aug_acc_means, aug_acc_stds, aug_frac_means = [], [], []
        p_means, p_stds, r_means, r_stds, f_means, f_stds = [],[],[],[],[],[]

        fracs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        if data_source in ("webapps", 'askubuntu'):
            fracs = fracs[:-2]
        if aug_algo == 'none':
            fracs = [1] + fracs

        # fracs = [0]

        for FRAC in fracs:
            statistics = repeat_augment_and_train(dir_to_save, data_source, aug_algo, args.encoder_model, args.sim_measure,
                                        (train_ds, val_ds, test_ds), TEXT, LABEL, FRAC, C, classifier_params, k=10)
            class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, aug_frac_mean, p_mean, p_std, r_mean, r_std, f_mean, f_std = statistics
            class_acc_means.append(class_acc_mean); class_acc_stds.append(class_acc_std)
            aug_acc_means.append(aug_acc_mean); aug_acc_stds.append(aug_acc_std); aug_frac_means.append(aug_frac_mean)
            p_means.append(p_mean); r_means.append(r_mean); f_means.append(f_mean)
            p_stds.append(p_std); r_stds.append(r_std); f_stds.append(f_std)
        
        for n, s in [('fracs', fracs), ('class_acc_means', class_acc_means), ('class_acc_stds', class_acc_stds), ('aug_acc_means', aug_acc_means),
                    ('aug_acc_stds', aug_acc_stds), ('aug_frac_means', aug_frac_means), ('p_means', p_means), ('p_stds', p_stds), ('r_means', r_means),
                    ('r_stds', r_stds), ('f1_means', f_means), ('f1_stds', f_stds)]:
            print(f"'{n}': {s},")

    # PLOT RESULTS
    elif args.task.startswith("plot"):
        data_source = args.task.split('_')[1]
        classifier = {'chatbot': 'nb', 'askubuntu': 'nb', 'webapps': 'sgd', 'chat': 'pool_max'}[data_source]

        if args.subtask == "vs_baseline":
            ss_methods = { # possible colors: c, b, g, m, y, k
                
                # BASELINE
                # "self-feed": {'algorithm': 'self_feed', 'colour': 'k'},
                
                # KNN-BASE
                # "KNN_b-GLoVe": {'algorithm': 'knn_base', 'encoder': 'glove', 'similarity': 'cosine', 'colour': 'c'},
                # "KNN_b-ELMo": {'algorithm': 'knn_base', 'encoder': 'elmo', 'similarity': 'cosine', 'colour': 'b'},
                # "KNN_b-Bert": {'algorithm': 'knn_base', 'encoder': 'bert', 'similarity': 'cosine', 'colour': 'g'}, # BEST
                # "KNN_b-InferSent": {'algorithm': 'knn_base', 'encoder': 'infersent', 'similarity': 'cosine', 'colour': 'y'},
                # "KNN_b-STS_sick": {'algorithm': 'knn_base', 'encoder': 'sts_sick', 'similarity': 'cosine', 'colour': 'c'},
                # "KNN_b-STS_stsbenchmark": {'algorithm': 'knn_base', 'encoder': 'sts_stsbenchmark', 'similarity': 'cosine', 'colour': 'b'},
                # "KNN_b-STS_both": {'algorithm': 'knn_base', 'encoder': 'sts_both', 'similarity': 'cosine', 'colour': 'y'},

                # KNN-Threshold
                # "KNN_t-GLoVe": {'algorithm': 'knn_threshold', 'encoder': 'glove', 'similarity': 'cosine', 'colour': 'c'},
                # "KNN_t-ELMo": {'algorithm': 'knn_threshold', 'encoder': 'elmo', 'similarity': 'cosine', 'colour': 'b'},
                # "KNN_t-Bert": {'algorithm': 'knn_threshold', 'encoder': 'bert', 'similarity': 'cosine', 'colour': 'b'}, # BEST
                # "KNN_t-InferSent": {'algorithm': 'knn_threshold', 'encoder': 'infersent', 'similarity': 'cosine', 'colour': 'y'},
                # "KNN_t-STS_sick": {'algorithm': 'knn_threshold', 'encoder': 'sts_sick', 'similarity': 'cosine', 'colour': 'c'},
                # "KNN_t-STS_stsbenchmark": {'algorithm': 'knn_threshold', 'encoder': 'sts_stsbenchmark', 'similarity': 'cosine', 'colour': 'b'},
                # "KNN_t-STS_both": {'algorithm': 'knn_threshold', 'encoder': 'sts_both', 'similarity': 'cosine', 'colour': 'c'},
                
                # LP-Base
                # "LP_b-GLoVe": {'algorithm': 'lp_base', 'encoder': 'glove', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_b-ELMo": {'algorithm': 'lp_base', 'encoder': 'elmo', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_b-Bert": {'algorithm': 'lp_base', 'encoder': 'bert', 'similarity': 'cosine', 'colour': 'g'}, # BEST
                # "LP_b-InferSent": {'algorithm': 'lp_base', 'encoder': 'infersent', 'similarity': 'cosine', 'colour': 'y'},
                # "LP_b-STS_sick": {'algorithm': 'lp_base', 'encoder': 'sts_sick', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_b-STS_sick (STS)": {'algorithm': 'lp_base', 'encoder': 'sts_sick', 'similarity': 'sts', 'colour': 'c'},
                # "LP_b-STS_stsbenchmark": {'algorithm': 'lp_base', 'encoder': 'sts_stsbenchmark', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_b-STS_stsbenchmark (STS)": {'algorithm': 'lp_base', 'encoder': 'sts_stsbenchmark', 'similarity': 'sts', 'colour': 'b'},
                # "LP_b-STS_both": {'algorithm': 'lp_base', 'encoder': 'sts_both', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_b-STS_both (STS)": {'algorithm': 'lp_base', 'encoder': 'sts_both', 'similarity': 'sts', 'colour': 'y'},

                # LP-Threshold
                # "LP_t-GLoVe": {'algorithm': 'lp_threshold', 'encoder': 'glove', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_t-ELMo": {'algorithm': 'lp_threshold', 'encoder': 'elmo', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_t-Bert": {'algorithm': 'lp_threshold', 'encoder': 'bert', 'similarity': 'cosine', 'colour': 'm'}, # BEST?
                # "LP_t-InferSent": {'algorithm': 'lp_threshold', 'encoder': 'infersent', 'similarity': 'cosine', 'colour': 'g'},
                # "LP_t-STS_sick": {'algorithm': 'lp_threshold', 'encoder': 'sts_sick', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_t-STS_sick (STS)": {'algorithm': 'lp_threshold', 'encoder': 'sts_sick', 'similarity': 'sts', 'colour': 'c'},
                # "LP_t-STS_stsbenchmark": {'algorithm': 'lp_threshold', 'encoder': 'sts_stsbenchmark', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_t-STS_stsbenchmark (STS)": {'algorithm': 'lp_threshold', 'encoder': 'sts_stsbenchmark', 'similarity': 'sts', 'colour': 'b'},
                # "LP_t-STS_both": {'algorithm': 'lp_threshold', 'encoder': 'sts_both', 'similarity': 'cosine', 'colour': 'm'},
                # "LP_t-STS_both (STS)": {'algorithm': 'lp_threshold', 'encoder': 'sts_both', 'similarity': 'sts', 'colour': 'y'},

                # LP-Recursive
                # "LP_r-GLoVe": {'algorithm': 'lp_recursive', 'encoder': 'glove', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_r-ELMo": {'algorithm': 'lp_recursive', 'encoder': 'elmo', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_r-Bert": {'algorithm': 'lp_recursive', 'encoder': 'bert', 'similarity': 'cosine', 'colour': 'c'}, # BEST OVERALL
                "LP_r-InferSent": {'algorithm': 'lp_recursive', 'encoder': 'infersent', 'similarity': 'cosine', 'colour': 'y'},
                # "LP_r-STS_sick": {'algorithm': 'lp_recursive', 'encoder': 'sts_sick', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_r-STS_sick (STS)": {'algorithm': 'lp_recursive', 'encoder': 'sts_sick', 'similarity': 'sts', 'colour': 'c'},
                # "LP_r-STS_stsbenchmark": {'algorithm': 'lp_recursive', 'encoder': 'sts_stsbenchmark', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_r-STS_stsbenchmark (STS)": {'algorithm': 'lp_recursive', 'encoder': 'sts_stsbenchmark', 'similarity': 'sts', 'colour': 'b'},
                # "LP_r-STS_both": {'algorithm': 'lp_recursive', 'encoder': 'sts_both', 'similarity': 'cosine', 'colour': 'g'}, # BEST OVERALL
                # "LP_r-STS_both (STS)": {'algorithm': 'lp_recursive', 'encoder': 'sts_both', 'similarity': 'sts', 'colour': 'y'},

                # LP-P1NN
                # "LP_p1nn-GLoVe": {'algorithm': 'lp_p1nn', 'encoder': 'glove', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_p1nn-ELMo": {'algorithm': 'lp_p1nn', 'encoder': 'elmo', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_p1nn-Bert": {'algorithm': 'lp_p1nn', 'encoder': 'bert', 'similarity': 'cosine', 'colour': 'k'},
                # "LP_p1nn-InferSent": {'algorithm': 'lp_p1nn', 'encoder': 'infersent', 'similarity': 'cosine', 'colour': 'y'},
                # "LP_p1nn-STS_sick": {'algorithm': 'lp_p1nn', 'encoder': 'sts_sick', 'similarity': 'cosine', 'colour': 'c'},
                # "LP_p1nn-STS_sick (STS)": {'algorithm': 'lp_p1nn', 'encoder': 'sts_sick', 'similarity': 'sts', 'colour': 'c'},
                # "LP_p1nn-STS_stsbenchmark": {'algorithm': 'lp_p1nn', 'encoder': 'sts_stsbenchmark', 'similarity': 'cosine', 'colour': 'b'},
                # "LP_p1nn-STS_stsbenchmark (STS)": {'algorithm': 'lp_p1nn', 'encoder': 'sts_stsbenchmark', 'similarity': 'sts', 'colour': 'b'},
                # "LP_p1nn-STS_both": {'algorithm': 'lp_p1nn', 'encoder': 'sts_both', 'similarity': 'cosine', 'colour': 'k'},
                # "LP_p1nn-STS_both (STS)": {'algorithm': 'lp_p1nn', 'encoder': 'sts_both', 'similarity': 'sts', 'colour': 'y'},
            }
            to_plot = ['f1']
            title = 'Ask Ubuntu: DOOT DOOT'
            plot_against_supervised(ss_methods, data_source, classifier, get_results, to_plot=to_plot, title=title, display=True, save_file=None)
        elif args.subtask == "stats":
            # plot all statistics for a given method (NB: in NLU datasets, weighted_recall=accuracy => don't print both)
            method = {"algorithm": "lp_recursive", 'encoder': 'sts_both', 'similarity': 'cosine'}
            statistics = [("f1", 'b'), ("p", 'y'), ("r", 'r')] # statistic, color
            plot_statistics(method, data_source, classifier, get_results, statistics, display=True, save_file=None)

    # INTENT CLASSIFICATION
    elif args.task.startswith("ic"):
        t = args.task.split('_'); task = t[0]; data_source = t[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        BS = 64 if args.encoder_model.startswith("pool") else 128
        EMB_DIM, HID_DIM = 300, 100
        FRAC = args.frac
        LR = 6e-4
        layers, drops = [EMB_DIM, HID_DIM, C], [0, 0]

        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        if args.encoder_model.startswith('pool'):
            encoder_args = [args.encoder_model.split('_')[1]]
        elif args.encoder_model == "lstm":
            encoder_args = []

        if args.subtask == "grid_search":
            if data_source == "snipsnlu":
                param_grid = {'lr': [1e-3, 6e-4, 3e-3, 6e-3], 'drop1': [0, 0.1], 'drop2': [0, 0.1], 'bs': [128, 64]}
            elif data_source == "chat":
                param_grid = {'lr': [3e-3, 6e-3, 1e-2], 'drop1': [0, 0.1, 0.2], 'drop2': [0, 0.1, 0.2], 'bs': [128]}
            results = grid_search(f'{args.saved_models}/chat_ic', train_ds, val_ds, test_ds, param_grid,
                            args.encoder_model, layers, TEXT, LABEL, k=5, verbose=False)
            print(results)
        elif args.subtask == "repeat_train":
            loss_func = nn.CrossEntropyLoss()
            datasets = (train_ds, val_ds, test_ds)
            dir_to_save = f'{args.saved_models}/{task}/{data_source}'
            mean, std = repeat_trainer(args.model_name, args.encoder_model, get_ic_data_iterators, IntentWrapper, dir_to_save, loss_func, datasets,
                                TEXT, LABEL, BS, encoder_args, layers, drops, LR, FRAC, k=10, verbose=True)
            print(f'Fraction of training data: {FRAC}, mean: {mean}, standard deviation: {std}')
        elif args.subtask == "train":
            train_di, val_di, test_di = get_ic_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            loss_func = nn.CrossEntropyLoss()
            wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/{task}/{data_source}',EMB_DIM,TEXT.vocab,args.encoder_model,train_di,val_di,test_di,encoder_args,layers=layers,drops=drops)
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=0)
            train_losses, val_losses = wrapper.train(loss_func, opt_func)

    # SEMANTIC TEXTUAL SIMILARITY
    elif args.task.startswith("sts"):
        C = 1
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.float, use_vocab=False)
        BS = 128
        LR = 6e-4
        EMB_DIM = 300
        HID_DIM = 300
        layers = [2*EMB_DIM, 300, C]
        drops = [0, 0]
        
        data_source = args.task.split('_')[1]
        saved_models_path = f'{args.saved_models}/sts/{data_source}'
        train_file = f'./data/sts/{data_source}/train_tknsd1.pkl' # './data/sts/both_train_tknsd.pkl'
        # train_file = './data/sts/both_train_tknsd.pkl'
        # val_file = f'./data/sts/{data_source}/val_tknsd.pkl'
        test_file = f'./data/sts/{data_source}/test_tknsd1.pkl' # './data/sts/both_test_tknsd.pkl'
        train_ds, val_ds, test_ds = STSDataReader(train_file, test_file, test_file, TEXT, LABEL).read() # using test data for validation
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        if args.encoder_model.startswith('pool'):
            encoder_args = [encoder_model.split('_')[1]] # pool_type
        elif args.encoder_model == 'lstm':
            encoder_args = [EMB_DIM, 1, False, False] # hidden_size, num_layers, bidirectional, fine_tune_we
        else:
            print(f'there are no classes set up for encoder model "{args.encoder_model}"')
        
        if args.subtask == "train":
            train_di, val_di, test_di = get_sts_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            loss_func = nn.MSELoss()
            wrapper = STSWrapper(args.model_name, saved_models_path, EMB_DIM, TEXT.vocab, args.encoder_model, encoder_args, args.predictor_model, layers, drops, train_di, val_di, test_di)
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=0)
            train_losses, val_losses, correlations = wrapper.train(loss_func, opt_func)
        elif args.subtask == "test":
            train_di, val_di, test_di = get_sts_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            wrapper = STSWrapper(args.model_name,saved_models_path,EMB_DIM,TEXT.vocab,args.encoder_model,args.predictor_model,train_di,val_di,test_di,layers,drops,*encoder_args)
            p,s = wrapper.test_correlation(load=True)
            print(f'pearson: {round(p[0],3)}, spearman: {round(s[0],3)}')



    elif args.task == "infersent":
        intents = pickle.load(Path('./data/chat_ic/intents.pkl').open('rb'))
        C = len(intents)
        TEXT = data.Field(sequential=True, use_vocab=False) # REMEMBER TO CHANGE THIS BACK IF I NEED TO.
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        BS = 128
        FRAC = args.frac
        LR = 6e-4
        layers = [300, 100, C]
        drops = [0, 0]

        import nltk
        from pretrained_models.infersent.models import InferSent
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        model = InferSent(params_model)
        model.load_state_dict(torch.load('./pretrained_models/infersent/infersent1.pkl'))
        model.set_w2v_path('./data/glove.840B.300d.txt')
        
        # build vocab
        train_ds, val_ds, test_ds = IntentClassificationDataReader('./data/chat_ic/', '_tknsd.pkl', TEXT, LABEL).read()
        sentences = [' '.join(eg.x) for eg in train_ds.examples]
        model.build_vocab(sentences, tokenize=False)

        # print(model.word_vec.keys())
        # I'll have to build a vocab here to satisfy Torchtext, and then use itos to convert from the
        # integers to the words before passing into InferSent?? this sounds like a massive pain in the ass...
        # but seems like it might be the best way... i.e. the model itself has a 'vocab' which is really just
        # its word2vec, and then TEXT also has a VOCAB that's used to create the iterator of the right size etc. etc.
        # I could try using a normal iterator (i.e. not bucket, setting the sort key param)???
        # try this before implementing the massive redundant thing

        # train
        train_di, val_di, test_di = get_ic_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
        print(next(iter(train_di)))
        print(next(iter(val_di)))
        print(next(iter(test_di)))
        # need to see if this is still text tokens rather than token indices... we don't want TEXT to create a vocab here.
        assert(False)
        loss_func = nn.CrossEntropyLoss()
        wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/chat_ic', 300, model, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
        opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=0)
        train_losses, val_losses = wrapper.train(loss_func, opt_func)


    elif args.task == "train_sts_benchmark":
        # import pandas as pd
        # df = pd.read_csv('./data/stsbenchmark/sts-train.csv', index_col=None, sep='\t', header=None, names=['to_delete', 'to_delete', 'to_delete', 'id', 'similarity', 's1', 's2'])
        # df = df.drop(columns=['to_delete', 'to_delete.1', 'to_delete.2', 'id'])
        # df.drop_duplicates()
        # df.reset_index(drop=True)
        # df = df[pd.notnull(df['s2'])]

        # import csv
        # test_data = []
        # with open('./data/stsbenchmark/sts-test.csv', 'r') as f:
        #     csv_reader = csv.reader(f)
        #     for r in csv_reader:
        #         row = list(filter(None, r))
        #         if len(row) == 1:
        #             test_data.append(row[0].split('\t'))``
        #         else:
        #             test_data.append(''.join(row).split('\t'))
        # test_data = [r[4:7] for r in test_data]

        # sentences = []
        # for row in df.iterrows():
        #     sentences += [row[1]['s1'], row[1]['s2']]
        # for row in test_data:
        #     sentences += [row[1], row[2]]
        # pickle.dump(sentences, Path('./data/stsbenchmark/sentences.pkl').open('wb'))

        # import nltk
        # from pretrained_models.infersent.models import InferSent
        # params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
        #                 'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        # model = InferSent(params_model)
        # model.load_state_dict(torch.load('infersent1.pkl'))
        # model.set_w2v_path('./data/glove.840B.300d.txt')
        # model.build_vocab(sentences, tokenize=True)
        # embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
        # pickle.dump(embeddings, Path('./data/stsbenchmark/embeddings.pkl').open('wb'))

        # sentences = pickle.load(Path('./data/stsbenchmark/sentences.pkl').open('rb'))
        # embeddings = pickle.load(Path('./data/stsbenchmark/embeddings.pkl').open('rb'))
        # s_enc_map = {}
        # for i, s in enumerate(sentences):
        #     if s not in s_enc_map:
        #         s_enc_map[s] = embeddings[i]
        # pickle.dump(s_enc_map, Path('./data/stsbenchmark/encoding_map.pkl').open('wb'))
        
        # enc = pickle.load(Path('./data/stsbenchmark/encoding_map.pkl').open('rb'))
        # train_data = []
        # for x in df.iterrows():
        #     train_data.append([enc[x[1]['s1']], enc[x[1]['s2']], [float(x[1]['similarity'])]])
        # pickle.dump(np.array(train_data), Path('./data/stsbenchmark/train_infersent.pkl').open('wb'))
        # test_data = np.array([[enc[x[1]], enc[x[2]], [float(x[0])]] for x in test_data])
        # pickle.dump(test_data, Path('./data/stsbenchmark/test_infersent.pkl').open('wb'))

        # train_data_1 = pickle.load(Path('./data/stsbenchmark/train_infersent.pkl').open('rb'))
        # train_data_2 = pickle.load(Path('./data/sick/train_infersent_2.pkl').open('rb'))
        # train_data = np.concatenate([train_data_1, train_data_2])
        # pickle.dump(train_data, Path('./data/sts_train_all.pkl').open('wb'))
        # test_data_1 = pickle.load(Path('./data/stsbenchmark/test_infersent.pkl').open('rb'))
        # test_data_2 = pickle.load(Path('./data/sick/test_infersent_2.pkl').open('rb'))
        # test_data = np.concatenate([test_data_1, test_data_2])
        # pickle.dump(test_data, Path('./data/sts_test_all.pkl').open('wb'))

        # train_di = STSDataIterator('./data/stsbenchmark/train_infersent.pkl', batch_size=50, randomise=True)
        # test_di = STSDataIterator('./data/stsbenchmark/test_infersent.pkl', randomise=False)
        # train_di = STSDataIterator('./data/sts_train_all.pkl', batch_size=50, randomise=True)
        # test_di = STSDataIterator('./data/sts_test_all.pkl', randomise=False)
        train_di = STSDataIterator('./data/sick/train_infersent_2.pkl', batch_size=50, randomise=True)
        test_di = STSDataIterator('./data/sick/test_infersent_2.pkl', randomise=False)
        layers, drops = [2*4096, 1024, 1], [0.3, 0]
        predictor = STSWrapper(args.model_name, args.saved_models, train_di, test_di, "pretrained", layers=layers, drops=drops)
        loss_func = nn.MSELoss()
        # opt_func = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=0, amsgrad=False)
        # opt_func = torch.optim.Adam(predictor.model.parameters(), lr=0.01, weight_decay=0, amsgrad=False)
        opt_func = torch.optim.SGD(predictor.model.parameters(), lr=0.01)
        train_losses, test_losses = predictor.train(loss_func, opt_func, 50)




    elif args.task == "train_sts_sick":

        # sentences = []
        # for s1,s2,_ in train_data_raw:
        #     sentences.append(s1)
        #     sentences.append(s2)
        # for s1,s2,_ in test_data_raw:
        #     sentences.append(s1)
        #     sentences.append(s2)
        # pickle.dump(sentences, Path('sentences.pkl').open('wb'))

        # import nltk
        # from pretrained_models.infersent.models import InferSent
        # params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
        #                 'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        # model = InferSent(params_model)
        # model.load_state_dict(torch.load('infersent1.pkl'))
        # model.set_w2v_path('./data/glove.840B.300d.txt')
        # model.build_vocab(sentences, tokenize=True)
        # embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
        
        # pickle.dump(sentences, Path('9k_sentences.pkl').open('wb'))
        # pickle.dump(embeddings, Path('embeddings.pkl').open('wb'))
        

        # sentences = pickle.load(Path('sentences.pkl').open('rb'))
        # embeddings = pickle.load(Path('embeddings.pkl').open('rb'))
        # s_enc_map = {}
        # for i, s in enumerate(sentences):
        #     if s not in s_enc_map:
        #         s_enc_map[s] = embeddings[i]
        # pickle.dump(s_enc_map, Path('encoding_map.pkl').open('wb'))
        
        # encoding_map = pickle.load(Path('encoding_map.pkl').open('rb'))


        train_di = STSDataIterator('./data/sts/sick/train_infersent_2.pkl', batch_size=64, randomise=True)
        test_di = STSDataIterator('./data/sts/sick/test_infersent_2.pkl', randomise=False)
        layers, drops = [2*4096, 512, 1], [0, 0, 0]
        wrapper = STSWrapper(args.model_name, args.saved_models, train_di, test_di, "pretrained", predictor_model="nn", layers=layers, drops=drops)
        loss_func = nn.MSELoss()
        opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=6e-4, weight_decay=0, amsgrad=False)
        train_losses, test_losses = wrapper.train(loss_func, opt_func, 50)


    elif args.task == "elmo":
        
        sentences = ['hey my name is Nick and I have a penis']
        
        # from mosestokenizer import MosesTokenizer, MosesDetokenizer
        # tokeniser = MosesTokenizer()
        # tknzd = tokeniser(sentences[0])
        # tokeniser.close()
        # print(tknzd)
        
        import spacy
        nlp = spacy.load('en')
        tknzd = [[t.text for t in nlp(s)] for s in sentences]

        from allennlp.commands.elmo import ElmoEmbedder
        from allennlp.modules.elmo import Elmo, batch_to_ids
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        
        # elmo = ElmoEmbedder(options_file, weight_file)
        # embeddings = elmo.embed_sentence(tknzd[0])
        # print(embeddings.shape)

        elmo = Elmo(options_file, weight_file, 1, dropout=0)
        character_ids = batch_to_ids(tknzd)
        embeddings = elmo(character_ids)
        print(embeddings['elmo_representations'][0].shape)
        # assuming I do one sentence at a time, this gives [seq_len, 1024] and then I can pool over these?

    elif args.task == "worst_predictions":
        predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, train_di, test_di, args.encoder_model, layers=layers, drops=drops)
        worst_predictions(predictor, test_data_raw, k=10)

    elif args.task == "nearest_neighbours":
        encoder = create_encoder(embedding_dim, args.encoder_model)
        encoder.load_state_dict(torch.load(f'{args.saved_models}/{args.model_name}_encoder.pt'))
        encoder.eval()
        s1 = "A woman is slicing potatoes"
        s2 = "Two men are playing guitar"
        s3 = "A boy is waving at some young runners from the ocean"
        nearest_neighbours(encoder, test_di, test_data_raw, [s1, s2, s3], k=20)

    elif args.task == "visualise_encoding":
        encoder = create_encoder(embedding_dim, args.encoder_model)
        encoder.load_state_dict(torch.load(f'{args.saved_models}/{args.model_name}_encoder.pt'))
        encoder.eval()
        encoder.evaluate = True
        s1, s2, _ = test_data_raw[381]
        st1, st2, _ = test_di.data[381]
        if args.encoder_model == "pos_lin":
            create_pos_lin_visualisations('./visualisations/s5_pos_lin', s1, st1, encoder)
            create_pos_lin_visualisations('./visualisations/s6_pos_lin', s2, st2, encoder)
        elif args.encoder_model == "pos_tree":
            create_pos_tree_visualisations('./visualisations/s5_pos_tree', s1, st1, encoder)
            create_pos_tree_visualisations('./visualisations/s6_pos_tree', s2, st2, encoder)


    # elif args.task == "test_sst":
    #     train_data = f'./data/sst/train_data_{args.word_embedding}_og.pkl'
    #     test_data = f'./data/sst/test_data_{args.word_embedding}_og.pkl's
    #     loss_func = nn.CrossEntropyLoss()
    #     layers = [embedding_dim, 500, 5]
    #     drops = [0, 0]
    #     encoder = create_encoder(embedding_dim, args.encoder_model)
    #     classifier = DownstreamWrapper(args.model_name, args.saved_models, "sst_classification", train_data, test_data, encoder, layers, drops)
    #     opt_func = torch.optim.Adagrad(classifier.model.parameters(), lr=args.lr, weight_decay=0)
    #     classifier.train(loss_func, opt_func, 15)
    #     classifier.save()
    #     print(classifier.test_accuracy())

    # elif args.task.startswith("probe"):
    #     probing_task = args.task.split("_", 1)[1]
    #     train_data = f'./data/senteval_probing/{probing_task}_train_tree.pkl'
    #     test_data = f'./data/senteval_probing/{probing_task}_test_tree.pkl'
    #     loss_func = nn.CrossEntropyLoss()
    #     layers = [embedding_dim, 200, 6]
    #     drops = [0, 0]
    #     encoder = create_encoder(embedding_dim, args.encoder_model)
    #     wrapper = ProbingWrapper(args.model_name, args.saved_models, probing_task, train_data, test_data, encoder, layers, drops)
    #     opt_func = torch.optim.SGD(wrapper.model.parameters(), lr=args.lr, weight_decay=0)
    #     wrapper.train(loss_func, opt_func, 10)
    #     wrapper.save()
    #     print(wrapper.test_accuracy())
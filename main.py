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

from results import all_results


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--subtask', type=str, default='none', help='sub-task within whichever specified task')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--model_name', type=str, default='999', help='name of model to train')
parser.add_argument('--encoder_model', type=str, default='pos_tree', help='encoder model for primary task')
parser.add_argument('--frac', type=float, default=1, help='fraction of training data to use')
parser.add_argument('--predictor_model', type=str, default='mlp', help='mlp / cosine_sim')
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
    if algorithm is 'supervised':
        results_name = f'{data_source}__supervised__{classifier}'
    else:
        results_name = f'{data_source}__{algorithm}__{encoder}__{similarity_measure}__{classifier}'
    
    return all_results[results_name]


####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################


if __name__ == "__main__":

    # embedding_dim = 300
    # layers = [2*embedding_dim, 250, 1]
    # drops = [0, 0]
    # sick_train_data = './data/sts/sick/train_data'
    # sick_test_data = './data/sts/sick/test_data'
    # train_data_raw = pickle.load(Path('./data/sts/sick/train.pkl').open('rb'))
    # test_data_raw = pickle.load(Path('./data/sts/sick/test.pkl').open('rb'))s
    # di_suffix = {"pos_lin": "og", "pos_tree": "trees", "dep_tree": "trees"}
    # train_di = EasyIterator(f'{sick_train_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl')
    # test_di = EasyIterator(f'{sick_test_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl', randomise=False)

    if args.task.startswith("propagater"):
        # params
        t = args.task.split("_")
        task = t[0]; data_source = t[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        EMB_DIM = 300

        # create datasets/vocab
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        FRAC = args.frac
        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        # ################################################################################################
        # # finding best parameters for label propagation algorithm for a given fraction.
        # results = grid_search_lp(data_source, 'propagation', args.encoder_model, train_ds, TEXT, LABEL, 0.1)
        # print("BEST 5 PARAM SETTINGS:")
        # print(results[:5])
        # assert(False)
        # ################################################################################################

        # pool_max intent classifier for for each learning method.
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

        # run augmentation trials
        aug_algo = 'self_feed' # 'label_prop__propagation' # 'knn'|'label_propagation__['propagation'|'spreading']'|None
        dir_to_save = f'{args.saved_models}/ic/{data_source}'

        class_acc_means, class_acc_stds, aug_acc_means, aug_acc_stds = [],[],[],[]
        p_means, p_stds, r_means, r_stds, f1_means, f1_stds = [],[],[],[],[],[]
        aug_frac_means = []
        for FRAC in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
            # print(self_train(dir_to_save, (train_ds,val_ds,test_ds), TEXT, LABEL, FRAC, classifier_params, verbose=False))
            # assert(False)
            class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, p_mean, p_std, r_mean, r_std, f1_mean, f1_std, aug_frac_mean = repeat_augment_and_train(dir_to_save, data_source, aug_algo, args.encoder_model, (train_ds, val_ds, test_ds), TEXT, LABEL, FRAC, classifier_params, k=5)
            class_acc_means.append(class_acc_mean); class_acc_stds.append(class_acc_std)
            aug_acc_means.append(aug_acc_mean); aug_acc_stds.append(aug_acc_std)
            p_means.append(p_mean); r_means.append(r_mean); f1_means.append(f1_mean)
            p_stds.append(p_std); r_stds.append(r_std); f1_stds.append(f1_std)
            aug_frac_means.append(aug_frac_mean)
        
        for stat in (class_acc_means, class_acc_stds, aug_acc_means, aug_acc_stds, aug_frac_means, p_means, p_stds, r_means, r_stds, f1_means, f1_stds):
            print(stat)


    elif args.task == "plot_results":
        ss_methods = {
            # "knn-BERT": {
            #     "algorithm": "knn_all",
            #     "encoder": "bert",
            #     "similarity": "cosine",
            #     "colour": "b-"
            # },
            # "knn-ELMo": {
            #     "algorithm": "knn_all",
            #     "encoder": "elmo",
            #     "similarity": "cosine",
            #     "colour": "g-"
            # },
            # "knn-GloVe": {
            #     "algorithm": "knn_all",
            #     "encoder": "glove",
            #     "similarity": "cosine",
            #     "colour": "o-"
            # },
            "my_lp-GloVe": {
                "algorithm": "my_lp",
                "encoder": "glove",
                "similarity": "cosine",
                "colour": "yo-",
                'ecolour': 'y'
            },
            "my_lp-ELMo": {
                "algorithm": "my_lp",
                "encoder": "elmo",
                "similarity": "cosine",
                "colour": "bo-",
                'ecolour': 'b'
            },
            "self_feed": {
                "algorithm": "self_feed",
                "encoder": "",
                "similarity": "",
                "colour": "go-",
                "ecolour": "g"
            }
        }
        data_source = 'chat'
        classifier = "pool_max"
        to_plot = "class_acc"
        plot_against_supervised(ss_methods, data_source, classifier, get_results, to_plot=to_plot, display=True, save_file=None)


    elif args.task.startswith("propagate"):
        data_source = args.task.split('_')[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        BS = 64
        EMB_DIM = 300
        FRAC = args.frac

        # get labeled & unlabeled data iterators
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)
        # CONVERT VOCAB FROM OLD TO NEW HERE SOMEHOW... IF WE WANT TO BE ABLE TO USE OUR PRETRAINED MODELS THAT IS.
        # can I just override update itos sheit somehow and it should work automatically??
        # refer to ULMFiT stuff
        
        # COSINE SIMILARITY: (otherwise we need an entire predictor distance measure thing...)

        # create encoder
        if args.encoder_model.startswith('pool'):
            encoder_args = [args.encoder_model.split('_')[1]] # pool_type
        elif args.encoder_model == 'lstm':
            encoder_args = [EMB_DIM, 1, False, False] # hidden_size, num_layers, bidirectional, fine_tune_we
        elif args.encoder_model.startswith("infersent"):
            # batch_size, encoding_dim, pool_type, dropout, version, state_dict_path, w2v_path, sentences, fine_tune
            pool_type = args.encoder_model.split('_')[1]
            sd_path = './pretrained_models/infersent/infersent1.pkl'
            w2v_path = './data/glove.840B.300d.txt'
            sentences = [' '.join(eg.x) for eg in train_ds.examples]
            encoder_args = [BS, 2048, pool_type, 0, 1, sd_path, w2v_path, sentences, False]
        
        encoder = create_encoder(TEXT.vocab, EMB_DIM, args.encoder_model, *encoder_args)
        
        # ''' # Only need the following if we are loading a model that I have trained... i.e. purely not InferSent at the moment.
        # enc_state_dict_path = f'{args.saved_models}/sts/stsbenchmark/lstm_0.709_encoder.pt'
        # encoder.load_state_dict(torch.load(enc_state_dict_path))
        # encoder.eval(); encoder.training = False
        # # WE HAVE TO CONVERT WEIGHTS FROM THE EMBEDDING LAYER TO THAT NEEDED IN INTENT CLASSIFICATION TASK WON'T I... I.E.
        # # WE HAVE A NEW VOCAB HERE...
        # print(encoder.state_dict().keys())
        # assert(False)
        # '''

        propagation_method = args.task.split('_')[2] # knn, lp
        augmented_train_examples = augment_data_all(propagation_method, encoder, args.encoder_model, train_ds, FRAC, TEXT, LABEL, BS, True)
        print(len(augmented_train_examples))

            #################################
            # KMEANS CLUSTERING
            #################################

            # from sklearn.cluster import KMeans
            # clusterer = KMeans(n_clusters=C, random_state=0).fit(Xs)
            # labels = clusterer.labels_
            
            # # find the indices of all data points in each cluster
            # clusters = {i: [] for i in range(C)}
            # for i, l in enumerate(labels):
            #     clusters[l].append(i)

            # # find the most common true label of all data points in each cluster
            # cluster_labels = {}
            # for l, idxs in clusters.items():
            #     lst = [Ys[i] for i in idxs]
            #     cluster_labels[l] = max(set(lst), key=lst.count)

            # # get accuracy
            # converted_labels = [cluster_labels[l] for l in labels]
            # accuracies.append(np.sum(converted_labels == Ys) / len(Ys))

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


    elif args.task.startswith("sts"):
        C = 1
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.float, use_vocab=False)
        BS = 64
        LR = 6e-4
        EMB_DIM = 300
        HID_DIM = 300
        layers = [2*EMB_DIM, HID_DIM, C]
        drops = [0, 0]
        
        data_source = args.task.split('_')[1]
        saved_models_path = f'{args.saved_models}/sts/{data_source}'
        train_file = f'./data/sts/{data_source}/train_tknsd.pkl' # './data/sts/both_train_tknsd.pkl'
        # train_file = './data/sts/both_train_tknsd.pkl'
        val_file = f'./data/sts/{data_source}/val_tknsd.pkl'
        test_file = f'./data/sts/{data_source}/test_tknsd.pkl' # './data/sts/both_test_tknsd.pkl'
        train_ds, val_ds, test_ds = STSDataReader(train_file, test_file, test_file, TEXT, LABEL).read() # using test data for validation
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        if args.encoder_model.startswith('pool'):
            encoder_args = [encoder_model.split('_')[1]] # pool_type
        elif args.encoder_model == 'lstm':
            encoder_args = [EMB_DIM, 1, False, False] # hidden_size, num_layers, bidirectional, fine_tune_we
        elif args.encoder_model == "infersent":
            encoder_args = []
        else:
            print(f'there are no classes set up for encoder model "{args.encoder_model}"')
        
        if args.subtask == "train":
            train_di, val_di, test_di = get_sts_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            loss_func = nn.MSELoss()
            wrapper = STSWrapper(args.model_name,saved_models_path,EMB_DIM,TEXT.vocab,args.encoder_model,args.predictor_model,train_di,val_di,test_di,layers,drops,*encoder_args)
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
        #             test_data.append(row[0].split('\t'))
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
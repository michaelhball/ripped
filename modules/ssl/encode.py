from modules.models import create_encoder
from modules.utilities.imports import *
from modules.utilities.imports_torch import *

__all__ = ['encode_data_with_pretrained']


def encode_data_with_pretrained(data_source, train_ds, text_field, encoder_model, examples_l, examples_u):
    if encoder_model.startswith('pretrained'):
        embedding_type = encoder_model.split('_')[1]
    else:
        embedding_type = encoder_model

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

    # normalise
    xs_l = np.array([x / np.linalg.norm(x) for x in xs_l])
    xs_u = np.array([x / np.linalg.norm(x) for x in xs_u])

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
        from flair.data import Sentence
        from flair.embeddings import BertEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings
        encoder = {"bert": BertEmbeddings(), "elmo": ELMoEmbeddings()}[embedding_type]
        sents = [Sentence(' '.join(eg.x)) for eg in train_ds.examples]
        encoder.embed(sents)
        embs = np.array([torch.max(torch.stack([t.embedding for t in S]), 0)[0].detach().numpy() for S in sents])
        embeddings = {' '.join(eg.x): emb for eg, emb in zip(train_ds.examples, embs)}

    pickle.dump(embeddings, Path(f'/Users/michaelball/Desktop/Thesis/repo/data/ic/{data_source}/{embedding_type}_embeddings.pkl').open('wb'))

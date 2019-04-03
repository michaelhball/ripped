"""
Class to handle training models using sklearn pipeline (for small datasets).
"""
import numpy as np

from pprint import pprint
from time import time
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

__all__ = ['do_basic_train_and_classify', 'do_grid_search']


vectorisers = {'count': CountVectorizer}
transformers = {'tfidf': TfidfTransformer}
classifiers = {'nb': MultinomialNB, 'lr': LogisticRegression, 'sgd': SGDClassifier, 'forest': RandomForestClassifier, 'xgboost': XGBClassifier}


def do_basic_train_and_classify(train_ds, test_ds, params, data_source, return_statistics=True):
    """
    Fits and classifies using an sklearn OOB model.
    Args:
        train_ds (Dataset): torchtext dataset of test data
        test_ds (Dataset): torchtext dataset of test data
        params (dict): all params needed to create pipeline; possible classifier params:
            nb: alpha
            lr: 
            sgd: loss (hinge|log|perceptron|squared_loss|...), penalty (none|l1|l2|elasticnet), alpha, learning_rate (constant|optimal|invscaling|adaptive)
            forest: 
            xgboost: 
    """
    encoder = LabelEncoder()
    train_text = [' '.join(eg.x) for eg in train_ds.examples]
    train_Y = encoder.fit_transform([eg.y for eg in train_ds.examples])
    test_text = [' '.join(eg.x) for eg in test_ds.examples]
    test_Y = encoder.fit_transform([eg.y for eg in test_ds.examples])

    classifier = Pipeline([
        ('vect', vectorisers[params['vectoriser']](**params['vect_args'])),
        ('tfidf', transformers[params['transformer']](**params['trans_args'])),
        ('clf', classifiers[params['classifier']](**params['class_args'])),
    ])

    classifier.fit(train_text, train_Y)
    if return_statistics:
        predictions = classifier.predict(test_text)
        acc = accuracy_score(predictions, test_Y)
        avg = "macro avg" if data_source == "chat" else "weighted avg"
        report = classification_report(predictions, test_Y, output_dict=True)[avg]
        p, r, f1 = report['precision'], report['recall'], report['f1-score']
        return acc, p, r, f1
    else: # means we're doing self-training
        if len(test_text) == 0: return [], []
        pred_probs = classifier.predict_proba(test_text)
        confidences = [np.max(p) for p in pred_probs]
        preds = [np.argmax(p) for p in pred_probs]
        return confidences, preds


def do_grid_search(train_ds):
    encoder = LabelEncoder()
    train_text = [' '.join(eg.x) for eg in train_ds.examples]
    train_Y = encoder.fit_transform([eg.y for eg in train_ds.examples])

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    params = {
        'vect__analyzer': ('word', 'char', 'char_wb'),
        'vect__binary': (True, False),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__ngram_range': ((1,1), (1,2), (2,2), (2,3)),
        'tfidf__norm': ('l1', 'l2'),
        'tfidf__use_idf': (True, False),
        'clf__alpha': (0.01, 0.1, 1),
    }

    grid_search = GridSearchCV(pipeline, params, cv=3, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(params)
    t0 = time()
    grid_search.fit(train_text, train_Y)
    print("done in %0.3fs" % (time.time() - t0))
    print("")
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    return best_parameters

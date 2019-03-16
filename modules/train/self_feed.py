from torchtext import data
from torchtext.data.example import Example

from modules.utilities import traina
from modules.utilities.imports import *
from modules.utilities.imports_torch import *

from .basic_trainer import do_basic_train_and_classify

__all__ = ['self_feed']


def self_feed(data_source, dir_to_save, iter_func, model_wrapper, labeled_examples, unlabeled_examples, val_ds, test_ds, text_field, label_field, classifier_params):
    """
    Implements a self-training algorithm.
    Args:
        data_source (str): Name of dataset we're using.
        dir_to_save (str): directory to save models created/loaded during this process
        labeled_examples (list(Example)): 
        unlabeled_examples (list(Example)): 
        text_field (Field): torchtext field for sentences
        label_field (LabelField): torchtext LabelField for class labels
        classifier_params (dict): all params needed to instantiate an intent classifier
    Returns:
        augmented labeled examples, accuracy of augmentation, fraction of unlabeled data used in augmentation
    """
    initial_labeled_num = len(labeled_examples)
    initial_unlabeled_num = len(unlabeled_examples)
    num_correct, num_added = 0, 0
    
    while True:
        new_train_ds = data.Dataset(labeled_examples, {'x': text_field, 'y': label_field})

        if data_source in ('askubuntu', 'chatbot', 'webapps'):
            test_ds_temp = data.Dataset(unlabeled_examples, {'x': text_field, 'y': label_field})
            confidences, preds = do_basic_train_and_classify(new_train_ds, test_ds_temp, classifier_params, return_statistics=False)
        else:
            ps = classifier_params
            args = [ps['model_name'], ps['encoder_model'], iter_func, model_wrapper, dir_to_save, nn.CrossEntropyLoss(),
                        (new_train_ds, val_ds, test_ds), text_field, ps['bs'], ps['encoder_args'], ps['layers'], ps['drops'], ps['lr']]
            wrapper = traina(*args, frac=1, verbose=False)
            confidences, preds = wrapper.classify_all(unlabeled_examples)

        new_unlabeled_examples = []
        for i, (conf, pred) in enumerate(zip(confidences, preds)):
            if conf > 0.9:
                labeled_examples.append(Example.fromdict({'x': unlabeled_examples[i].x, 'y': pred}, {'x': ('x', text_field), 'y': ('y', label_field)}))
                num_added += 1
                if pred == unlabeled_examples[i].y:
                    num_correct += 1
            else:
                new_unlabeled_examples.append(unlabeled_examples[i])

        if len(unlabeled_examples) - len(new_unlabeled_examples) == 0:
            break
        unlabeled_examples = new_unlabeled_examples
        np.random.shuffle(labeled_examples)
    
    frac_used = float(num_added/initial_unlabeled_num)
    aug_acc = 0 if num_added == 0 else float(num_correct/num_added)

    return labeled_examples, aug_acc, frac_used
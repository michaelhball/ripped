import time
import torch

from scipy.stats.stats import pearsonr, spearmanr
from tqdm import tqdm

from modules.models import create_multi_task_learner
from modules.utilities import V

from .base_wrapper import BaseWrapper


class MTLWrapper(BaseWrapper):
    """
    A class for training within mulit-task learning framework.
    """
    def __init__(self, name, saved_models, embedding_dim, encoder_model, sts_data, nli_data, sts_dims, nli_dims):
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.sts_train_di, self.sts_val_di, self.sts_test_di = sts_data
        self.nli_train_di, self.nli_val_di, self.nli_test_di = nli_data
        self.encoder_model = encoder_model
        self.sts_dims, self.nli_dims = sts_dims, nli_dims
        self.create_model()

    def save(self):
        self.save_model()
        self.save_encoder()

    def save_model(self):
        path = self.saved_models + f'/{self.name}.pt'
        torch.save(self.model.state_dict(), path)

    def save_encoder(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        torch.save(self.model.encoder.state_dict(), path)
    
    def save_sts_head(self):
        path = self.saved_models + f'/{self.name}_sts_head.pt'
        torch.save(self.model.sts_head.state_dict(), path)

    def save_nli_head(self):
        path = self.saved_models + f'/{self.name}_nli_head.pt'
        torch.save(self.model.nli_head.state_dict(), path)
    
    def load(self):
        path = self.saved_models + f'/{self.name}.pt'
        self.model.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        self.model.encoder.load_state_dict(torch.load(path))
    
    def create_model(self):
        self.model = create_multi_task_learner(self.embedding_dim, self.encoder_model, self.sts_dims, self.nli_dims)

    # def test_correlation(self, load=False):
    #     if load:
    #         self.create_model()
    #         self.load()
    #     self.model.eval()
    #     self.model.training = False
    #     preds, scores, info = [], [], []
    #     for i, example in enumerate(iter(self.test_di)):
    #         s1, s2, score = example
    #         pred = self.model(s1, s2)
    #         preds.append(pred.item())
    #         scores.append((score-1)/4)
    #         info.append((i, (score-1)/4, pred.item()))
        
    #     pearson = pearsonr(preds, scores)
    #     spearman = spearmanr(preds, scores)
    #     info = sorted(info, key=lambda tup: abs(tup[1]-tup[2]), reverse=True)
        
    #     return pearson, spearman, info
    
    # def avg_test_loss(self, loss_func):
    #     self.model.eval()
    #     self.model.training = False
    #     total_loss = 0.0
    #     for i, example in enumerate(iter(self.test_di)):
    #         s1, s2, score = example
    #         pred = self.model(s1, s2)
    #         total_loss += loss_func(pred[0], V((score)-1)/4)
        
    #     return total_loss / self.test_di.num_examples

    def construct_training_iterator(self, sts_di, nli_di):
        # needs to look at number_of_batches of each, 
        # and organise them so that we alternate accordingly.
        # e.g. if len(sts_di) = 2xlen(nli_di), we want to put 2
        # items from sts_di for every 1 from nli_di in an iterator
        # we basically just need to create a list right?
        
        

        pass

    def train(self, loss_func, sts_opt_func, nli_opt_func, num_epochs=50):
        print("-------------------------  Training Multi-Task Learner -------------------------")
        start_time = time.time()
        train_losses, val_losses = [], []
        opt_funcs = {"sts": sts_opt_func, "nli": nli_opt_func}
        loss_funcs = {"sts": sts_loss_func, "nli": nli_opt_func}

        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_losses = {"sts": 0.0, "nli": 0.0}
            
            self.training_iterator = self.construct_training_iterator(self.sts_train_di, self.nli_train_di)
            for i, example in tqdm(enumerate(iter(self.training_iterator)), total=len(self.training_iterator)):
                # each datum will be (task_type, (X, Y)) where the xs and ys are batched.
                task = example[0]
                opt_funcs[task].zero_grad()
                X, Y = example[1]
                preds = self.model(X)
                loss = loss_funcs[task](preds[0], V(Y))
                total_losses[task] += loss.item()
                loss.backward()
                opt_funcs[task].step()
            
            # NEED TO CALCULATE THESE FOR EACH TYPE OF LOSS
            # avg_train_loss = total_loss / self.train_di.num_examples
            # avg_test_loss = self.avg_test_loss(loss_func)
            # train_losses.append(avg_train_loss)
            # test_losses.append(avg_test_loss)

            # p, s, _ = self.test_correlation()
            # correlations.append((e+1, round(p[0],3), round(s[0],3)))
            # path = self.saved_models + f'/{self.name}_{e+1}.pt'
            # torch.save(self.model.state_dict(), path)
            # path = self.saved_models + f'/{self.name}_encoder_{e+1}.pt'
            # torch.save(self.model.encoder.state_dict(), path)

            # print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg test loss: {avg_test_loss}, test pearson: {round(p[0], 3)}")

        elapsed_time = time.time() - start_time
        print("Trained MTL completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, val_losses

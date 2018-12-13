import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from modules.utilities import my_dependencies, universal_tags


class DEPTree(nn.Module):
    def __init__(self, embedding_dim, evaluate=False):
        """
        Sentence embedding model using dependency parse structure.
        Args:
            embedding_dim (int): dimension of word/phrase/sentence embeddings
            evaluate (bool): Indicator as to whether we want to evaluate embeddings.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.evaluate = evaluate

        # self.dependency_map = my_dependencies
        # self.num_params = max(list(self.dependency_map.values())) + 1
        # self.params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_params)])
        # self.params = nn.ModuleList([nn.Linear(self.embedding_dim*2, self.embedding_dim) for _ in range(self.num_params)])

        self.dep_tags = my_dependencies
        self.pos_tags = universal_tags
        self.num_dep_params = max(list(self.dep_tags.values())) + 1
        self.num_pos_params = max(list(self.pos_tags.values())) + 1
        self.dep_params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_dep_params)])
        self.pos_params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_pos_params)])

        self.dep_freq = {}

    def recur(self, node):
        # x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        # z_cs = []
        # for c in node.chidren:
        #     x_c = self.recur(c)
        #     if c.dep not in self.dependency_map:
        #         continue
        #     D_c = self.params[self.dependency_map[c.dep]]
        #     z_cs.append(F.relu(D_c(x_c)))
        
        # zs = torch.stack(z_cs + [x])
        # z, _ = torch.max(zs, 0)

        # x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        # z_cs = []
        # for c in node.chidren:
        #     if c.dep not in self.dependency_map:
        #         continue
        #     x_c = self.recur(c)
        #     D_c = self.params[self.dependency_map[c.dep]]
        #     a = torch.cat((x, x_c)).view(1, -1)
        #     z_cs.append(F.relu(D_c(a).view(1,-1)))
        
        # if z_cs:
        #     z, _ = torch.max(torch.stack(z_cs), 0)
        #     z = z.view(1,-1)
        # else:
        #     z = x

        # # 356
        # x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        # x = F.relu(self.pos_params[self.pos_tags[node.pos]](x))
        # z_cs = []
        # for c in node.chidren:
        #     if c.dep not in self.dep_tags:
        #         continue
        #     x_c = self.recur(c)
        #     D_c = self.dep_params[self.dep_tags[c.dep]]
        #     z_c = F.relu(D_c(torch.cat((x,x_c)).view(1,-1)))
        #     # z_c = F.relu(D_c(torch.cat((x,x*x_c)).view(1,-1)))
        #     z_cs.append(z_c)
        
        # if z_cs:
        #     # z, _ = torch.max(torch.stack(z_cs), 0)
        #     z = torch.mean(torch.stack(z_cs), 0)
        # else:
        #     z = x

        # 357
        x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        x = F.relu(self.pos_params[self.pos_tags[node.pos]](x))
        z_cs = []
        for c in node.chidren:
            if c.dep not in self.dep_tags:
                continue
            x_c = self.recur(c)
            D_c = self.dep_params[self.dep_tags[c.dep]]
            z_c = F.relu(D_c(x_c))
            z_cs.append(z_c)
        
        z, _ = torch.max(torch.stack(z_cs+[x]), 0)
        # z = torch.mean(torch.stack(z_cs), 0)






        return z

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            output = self.recur(input)
            if self.evaluate:
                return input
            return output

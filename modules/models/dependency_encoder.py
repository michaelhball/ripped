import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .linear_block import LinearBlock


class DependencyEncoder(nn.Module):
    def __init__(self, embedding_dim, batch_size, dependency_map):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.dependency_map = dependency_map
        self.num_params = max(list(self.dependency_map.values())) + 1
        self.params = nn.ParameterList([nn.Parameter(torch.zeros(self.embedding_dim, self.embedding_dim)) for _ in range(self.num_params)])
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(self.num_params)])
        self._init_params()

        # to analyse dependency usage
        self.dep_freq = {}
        self.rare_dependencies = set()
    
    def _init_params(self):
        for param in self.params.parameters():
            nn.init.xavier_uniform_(param)
    
    def _is_leaf(self, node):
        return False if node.chidren else True

    def recur(self, node):
        x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        if self._is_leaf(node):
            z = x
        else:
            z_cs = []
            for c in node.chidren:
                if c.dep not in self.dep_freq:
                    self.dep_freq[c.dep] = 0
                else:
                    self.dep_freq[c.dep] += 1
                if c.dep not in self.dependency_map:
                    if c.dep not in self.rare_dependencies:
                        self.rare_dependencies.add(c.dep)
                    continue
                
                dep_index = self.dependency_map[c.dep]
                D_c = self.params[dep_index] # dxd
                x_c = self.recur(c).repeat(self.embedding_dim, 1) # dxd
                # x_c = self.dropouts[dep_index](self.recur(c).repeat(self.embedding_dim, 1))
                z_c = F.relu(x_c * D_c) # dxd
                z_cs.append(z_c)
            
            if not z_cs:
                z = x
            else:
                zs = torch.stack(z_cs) # nxdxd (n = #children)
                mult, _ = torch.max(zs, 0) # dxd
                z = torch.matmul(x, mult) # 1xd for output to next layer

        return z

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            return self.recur(input)






# class DependencyEncoder(nn.Module):
#     """
#     A dependency encoder based on spacy Dependency Parsing.
#     """
#     def __init__(self, nlp, word_embeddings, embedding_dim, batch_size, dependency_map, use_bias=False):
#         """
#         Args:
#             nlp ()
#         """
#         super().__init__()
#         self.nlp = nlp
#         self.we = word_embeddings
#         self.embedding_dim = embedding_dim
#         self.batch_size = batch_size
#         self.dependency_map = dependency_map
#         self.num_params = max(list(self.dependency_map.values())) + 1
#         self.use_bias = use_bias
#         self.params = nn.ParameterList([nn.Parameter(torch.zeros(self.embedding_dim, self.embedding_dim)) for _ in range(self.num_params)])
#         # self.params = nn.ModuleList([LinearBlock(self.embedding_dim, self.embedding_dim, 0.5) for _ in range(self.num_params)])
#         self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(self.num_params)]) # not sure if we need all of this like this?
#         if self.use_bias:
#             self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, self.embedding_dim)) for _ in range(self.num_params)])
#         self._init_params()
        
#         # for testing
#         self.dep_freq = {}
#         self.unseen_words = []
#         self.rare_dependencies = set()

#     def _init_params(self): # decide whether this is the best form of initialisation.
#         for param in self.params.parameters():
#             nn.init.xavier_uniform_(param)
#         if self.use_bias:
#             for bias in self.biases.parameters():
#                 nn.init.xavier_uniform_(bias)
    
#     def _is_leaf(self, node):
#         return False if list(node.children) else True

#     # def _node_embedding(self, node):
#     #     node_text = node.text.lower()
#     #     if node_text not in self.we:
#     #         self.unseen_words.append(node_text)
#     #         print("token <'{0}'> not in word embeddings, replaced with <unk>".format(node.text))
#     #         node_text = 'unk'

#     #     x = Variable(torch.tensor([self.we[node_text]], dtype=torch.float), requires_grad=True)

#     #     if self._is_leaf(node):
#     #         z = x
#     #     else:
#     #         z_cs = []
#     #         for c in node.children:
#     #             if c.dep_ not in self.dep_freq:
#     #                 self.dep_freq[c.dep_] = 0
#     #             else:
#     #                 self.dep_freq[c.dep_] += 1
#     #             if c.dep_ not in self.dependency_map:
#     #                 if c.dep_ not in self.rare_dependencies:
#     #                     self.rare_dependencies.add(c.dep_)
#     #                 # DO this after training, when testing, so I can see when these cases arise...
#     #                 # if c.dep_ not in self.rare_dependencies: # put entire Token in there so we can check out all dependencies later
#     #                 #     self.rare_dependencies[c.dep_] = [c]
#     #                 # else:
#     #                 #     self.rare_dependencies.append(c)
#     #                 continue
#     #             dep_index = self.dependency_map[c.dep_]
#     #             D_c = self.params[dep_index] # dxd
#     #             x_c = self._node_embedding(c) # 1xd
#     #             z_c = torch.matmul(x_c, D_c) # 1xd
#     #             if self.use_bias:
#     #                 b_c = self.biases[dep_index]
#     #                 z_c = torch.add(z_c, b_c)
#     #             z_c = torch.tanh(z_c) # 1xd
#     #             z_cs.append(z_c)

#     #         zs = torch.stack([x] + z_cs) # nx1xd (n = # children+1)
#     #         # z, _ = torch.max(zs, 0) # 1xd
#     #         z = torch.mean(zs, 0) # 1xd
            
#     #     return z
    
#     def _node_embedding_2(self, node):
#         node_text = node.text.lower()
#         if node_text not in self.we:
#             self.unseen_words.append(node_text)
#             # print("token <'{0}'> not in word embeddings".format(node.text))
#             node_text = 'unk'
            
#         x = Variable(torch.tensor([self.we[node_text]], dtype=torch.float), requires_grad=True)

#         if self._is_leaf(node):
#             z = x
#         else:
#             z_cs = [] # holds the multipliers of all children.
#             for c in node.children:
#                 if c.dep_ not in self.dep_freq:
#                     self.dep_freq[c.dep_] = 0
#                 else:
#                     self.dep_freq[c.dep_] += 1
#                 if c.dep_ not in self.dependency_map:
#                     if c.dep_ not in self.rare_dependencies:
#                         self.rare_dependencies.add(c.dep_)
#                     continue
                
#                 dep_index = self.dependency_map[c.dep_]
#                 D_c = self.params[dep_index]
#                 x_c = self.dropouts[dep_index](self._node_embedding_2(c).repeat(self.embedding_dim, 1))
#                 z_c = x_c * D_c
#                 if self.use_bias:
#                     z_c += self.biases[dep_index]
#                 z_c = torch.tanh(z_c)
#                 z_cs.append(z_c)
            
#             if not z_cs:
#                 z = x
#             else:
#                 zs = torch.stack(z_cs) # nxdxd (where n = number of chldren)
#                 # mult = torch.max(zs, 0) # dxd = elt-wise max of child multipliers
#                 mult = torch.mean(zs, 0) # dxd = elt-wise average of child multipliers
#                 z = torch.matmul(x, mult) # add bias or non-linearity here to final output of this nodes representation??
        
#         return z

#     def forward(self, input):
#         root = list(self.nlp(input).sents)[0].root

#         return self._node_embedding_2(root)

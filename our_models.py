import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GINConv, GATConv
import warnings


class GNN(torch.nn.Module):
    def __init__(self, embed, gnn_layers, gnn_type, device):
        super(GNN, self).__init__()
        h = embed

        def get_layer(gnn_type):
            if gnn_type == 'ChebConv':
                layer = ChebConv(h, h, K=2)
            elif gnn_type == 'GCNConv':
                layer = GCNConv(h, h)
            elif gnn_type == 'GINConv':
                dnn = nn.Sequential(nn.Linear(h, h),
                                    nn.LeakyReLU(),
                                    nn.Linear(h, h))
                layer = GINConv(dnn)
            elif gnn_type == 'SAGEConv':
                layer = SAGEConv(h, h, normalize=True)
            elif gnn_type == 'GATConv':
                layer = GATConv(h, h)
            else:
                raise NotImplementedError
            return layer

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        if gnn_layers >= 1:
            self.conv1 = get_layer(gnn_type)
        if gnn_layers >= 2:
            self.conv2 = get_layer(gnn_type)
        if gnn_layers == 3:
            self.conv3 = get_layer(gnn_type)

    def forward(self, embeddings, edge_index):

        for layer in [self.conv1, self.conv2, self.conv3]:
            if layer is not None:
                embeddings = layer(embeddings, edge_index)

        return embeddings


class SharedBilinearDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, num_relations, num_weights, embed_dim):
        super(SharedBilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_weights, embed_dim * embed_dim)
        self.weight_scalars = nn.Parameter(torch.Tensor(num_weights, num_relations))
        stdv = 1. / math.sqrt(self.weight_scalars.size(1))
        self.weight_scalars.data.uniform_(-stdv, stdv)
        self.embed_dim = embed_dim
        self.num_weights = num_weights
        self.num_relations = num_relations
        self.nll = nn.NLLLoss()
        self.mse = nn.MSELoss()

    def predict(self, embeds1, embeds2):
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, \
                                                     self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        outputs = F.log_softmax(logit, dim=1)
        preds = 0
        for j in range(0, self.num_relations):
            index = (torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            preds += (j + 1) * torch.exp(torch.index_select(outputs, 1, index))
        return preds

    def forward(self, embeds1, embeds2, rels):
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, \
                                                     self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        outputs = F.log_softmax(logit, dim=1)
        log_probs = torch.gather(outputs, 1, rels.unsqueeze(1))
        loss = self.nll(outputs, rels)
        preds = 0
        for j in range(0, self.num_relations):
            index = (torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            preds += (j + 1) * torch.exp(torch.index_select(outputs, 1, index))
        return loss, preds


class OurGAL(nn.Module):
    def __init__(self, decoder, embed_dim, num_ent, edges, args, encoder=None, hop=2):
        super(OurGAL, self).__init__(decoder, embed_dim, num_ent, edges, args, encoder=None)

        edges_np = edges.numpy()
        print(edges_np.max())
        edge_list = []
        for i in tqdm(range(edges_np.shape[1])):
            edge_list.append((edges_np[0, i], edges_np[1, i]))
        self.G = nx.Graph(edge_list)
        self.hop = hop

    def encode(self, nodes):
        embs = self.encoder(self.all_nodes)
        embs = self.batchnorm(embs)
        embs = self.gnn(embs, self.edges)
        if embs is None:
            return embs
        return embs[nodes]

    def forward_attr(self, user_features, weights=None, return_embeds=False, ):
        (users, gender, occupation, age) = user_features

        neighbor_sub = []

        user_np = list(users.cpu().numpy())

        k = 0
        include_indices = []
        ign_indices = []
        ign_str = "failed vertices: "
        for user in user_np:
            final = user
            path = dict()
            for _ in range(self.hop):
                neighbor = list(self.G.neighbors(final))
                L = lambda a: len(a)
                orig_len = L(neighbor)

                cond = True
                cand = neighbor.pop(np.random.randint(0, L(neighbor)))
                if (cand in path):

                    # This is bounded by O(self.hop)
                    while (L(neighbor) > 0):
                        cand = neighbor.pop(np.random.randint(0, L(neighbor)))
                        if (cand not in path):
                            break
                    if (L(neighbor) == 0):
                        ign_str += "[{} <= {}]".format(orig_len, len(path))
                        cond = False
                        break

                if cond:
                    path[cand] = 0
                    final = cand
                else:
                    break
            if cond:
                neighbor_sub.append(final)
                include_indices.append(k)
            else:
                ign_indices.append(k)
            k += 1

        include_indices = np.array(include_indices)
        if (len(ign_indices) > 0): warnings.warn("ignoring {} from {}".format(ign_indices, ign_str), RuntimeWarning,
                                                 stacklevel=2)
        gender = gender[include_indices]
        age = age[include_indices]
        occupation = occupation[include_indices]

        neighbor = torch.tensor(np.array(neighbor_sub)).cuda()

        user_embeds = self.reverse(self.encode(neighbor))

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if (self.mode == 'gender'):
            loss_adv = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss_adv = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss_adv = fn_occupation(occupation_pred, occupation)

        return loss_adv, (age_pred, gender_pred, occupation_pred)

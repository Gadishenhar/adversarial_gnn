from our_models import *
from argparse import ArgumentParser
from helper import *
from torch.utils.data import Dataset, DataLoader
from random import random
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GINConv, GATConv  # noqa
from torch_geometric.utils import train_test_split_edges
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, f1_score
from classes.basic_classes import GNN_TYPE, DataSet
from classes.attack_class import AttackMode
import sys
from argparse import ArgumentParser
from torch.cuda import set_device
import copy
from classes.basic_classes import DatasetType, DataSet

print(os.getcwd())

def main(args=None):
    if args is None:
        parser = ArgumentParser()
        parser.add_argument("-num_relations", dest="num_relations", default=5, required=False)
        parser.add_argument("-num_weights", dest="num_weights", default=2, required=False)
        parser.add_argument("-num_layers", dest="num_layers", default=2, required=False)
        parser.add_argument("-GAL_gnn_type", dest="GAL_gnn_type", default='GCNConv', required=False)
        parser.add_argument("-dataset", dest="dataset", default='pubmed', required=False)
        parser.add_argument("-batch", dest="batch", default=256, required=False)
        parser.add_argument("-lr", dest="lr", default=0.01, required=False)
        parser.add_argument("-use_gdc", dest="use_gdc", default=False, required=False)
        parser.add_argument("-num_epochs", dest="num_epochs", default=50, required=False)
        parser.add_argument("-finetune_epochs", dest="finetune_epochs", default=10, required=False)
        parser.add_argument("-attMode", dest="attMode", default=AttackMode.NODE, type=AttackMode.from_string,
                            choices=list(AttackMode), required=False)
        parser.add_argument('-singleGNN', dest="singleGNN", type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                            required=False)
        parser.add_argument("-patience", dest="patience", default=20, type=int, required=False)
        parser.add_argument("-attEpochs", dest="attEpochs", default=20, type=int, required=False)
        parser.add_argument("-l_inf", dest="l_inf", type=float, default=None, required=False)
        parser.add_argument('-targeted', dest="targeted", action='store_true', required=False)
        parser.add_argument("-distance", dest='distance', type=int, required=False)
        parser.add_argument("-seed", dest="seed", type=int, default=0, required=False)
        parser.add_argument('-gpu', dest="gpu", type=int, required=False)
        parser.add_argument("-lam", dest="lam", default=1.25, required=False)
        parser.add_argument("-SINGLE", dest="SINGLE", default=False, type=bool, required=False)
        args = parser.parse_args()
    # define the arguments for the attack
    att_args = copy.deepcopy(args)
    if args.dataset == 'pubmed':
        att_args.dataset = DataSet.PUBMED
    if args.dataset == 'citeseer':
        att_args.dataset = DataSet.CITESEER
    if args.dataset == 'cora':
        att_args.dataset = DataSet.CORA

    # create attack instance based on the arguments
    if args.SINGLE:
        attack = att_args.attMode.getAttack()
        attack = attack(att_args)
        attack_model = attack.defineWrapper(att_args)
    else:
        attack_model = None

    # Load the data set:
    if args.dataset == 'pubmed' or 'citeseer' or 'cora':
        print(f"current dataset is {args.dataset}")
        dataset_path = os.path.join(getGitPath(), 'datasets')
        dataset = Planetoid(dataset_path, args.dataset, transform=T.NormalizeFeatures())
        num_nodes = dataset.data.num_nodes
        num_edges = dataset.data.num_edges
        edges = dataset.data.edge_index
    train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True)


    # Define the defense model:
    num_hops = 2
    embed_dim = num_nodes
    decoder = SharedBilinearDecoder(args.num_relations, args.num_weights, embed_dim)
    defense_model = OurGAL(decoder, embed_dim, num_nodes, edges, args, encoder=None, hop=num_hops)  # passing
    # encoder=None makes the model use the default nn.Embedding encoder. This can also be done by:
    # encoder = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2)
    defense_optimizer = create_optimizer([
                                {'params': defense_model.encoder.parameters()},
                                {'params': defense_model.batchnorm.parameters()},
                                {'params': defense_model.decoder.parameters()},
                                {'params': defense_model.gnn.parameters()}], 'adam', args.lr)

    # GAL paper loops the following parameters. For our checking we will use first seed=100 lam=1.25 and m=GCNConv
    # for seed in [100, 200, 300, 400, 500]:
    #     for lam in [0, 1.25, 1.75]:
    #         for m in ['GINConv', 'GCNConv', 'GATConv', 'ChebConv']:

    lam = args.lam
    print(f"###################### LAMBDA = {lam} #########################")

    m = args.GAL_gnn_type
    seed = 100

    res = {}

    args.seed = seed
    args.lambda_reg = lam

    # define the result dictionary
    try:
        res[m][seed][lam] = {}
    except:
        try:
            res[m][seed] = {lam: {}}
        except:
            res[m] = {seed: {lam: {}}}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    data = dataset[0]

    # GDC approximation for the data:
    if args.use_gdc:
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=128,
                                               dim=0), exact=True)
        data = gdc(data)

    labels = data.y
    edge_index, edge_weight = data.edge_index, data.edge_attr

    print(labels.size())

    # Split into subsets:
    split_data = train_test_split_edges(data)

    # Declare the device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the net:
    model, data = Net(edge_index=edge_index, edge_weight=edge_weight, data=split_data, num_classes=dataset.num_classes,
                      attack=attack_model, name=m).to(device), data.to(device)

    if m == 'GINConv':
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.bn1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.bn2.parameters(), weight_decay=0),
        ], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)

    if m == 'GINConv':
        optimizer_att = torch.optim.Adam([
            dict(params=model.conv2.parameters(), weight_decay=5e-4),
            dict(params=model.bn2.parameters(), weight_decay=0),
            dict(params=model.attack.parameters(), weight_decay=5e-4),
        ], lr=args.lr * float(args.lambda_reg))
    else:
        optimizer_att = torch.optim.Adam([
            dict(params=model.conv2.parameters(), weight_decay=5e-4),
            dict(params=model.attack.parameters(), weight_decay=5e-4),
        ], lr=args.lr * float(args.lambda_reg))

    def get_link_labels(pos_edge_index, neg_edge_index):
        link_labels = torch.zeros(pos_edge_index.size(1) +
                                  neg_edge_index.size(1)).float().to(device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def train():
        model.train()
        optimizer.zero_grad()
        optimizer_att.zero_grad()
        x, pos_edge_index = data.x, data.train_pos_edge_index

        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                           num_nodes=x.size(0))

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1))

        link_logits, attr_prediction, attack_prediction = model(pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        losses = [F.binary_cross_entropy_with_logits(link_logits, link_labels), F.nll_loss(attack_prediction, labels)]
        loss = sum(losses)  # We take into consideration both losses by summing them up
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer_att.step()
        return loss

    def test():
        model.eval()
        perfs = []
        for prefix in ["val", "test"]:
            pos_edge_index, neg_edge_index = [
                index for _, index in data("{}_pos_edge_index".format(prefix),
                                           "{}_neg_edge_index".format(prefix))
            ]
            link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index)[0])
            link_labels = get_link_labels(pos_edge_index, neg_edge_index)
            link_probs = link_probs.detach().cpu().numpy()
            link_labels = link_labels.detach().cpu().numpy()
            perfs.append(roc_auc_score(link_labels, link_probs))
        return perfs

    best_val_perf = test_perf = 0
    for epoch in range(1, int(args.num_epochs) + 1):
        train_loss = train()
        val_perf, tmp_test_perf = test()
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
            res[m][seed][lam]['task'] = {'val': best_val_perf, 'test': test_perf}
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, val_perf, tmp_test_perf))

    optimizer_attr = torch.optim.Adam([
        dict(params=model.attr.parameters(), weight_decay=5e-4),
    ], lr=args.lr)

    def train_attr():
        model.train()
        optimizer_attr.zero_grad()

        x, pos_edge_index = data.x, data.train_pos_edge_index

        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                           num_nodes=x.size(0))

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
            num_neg_samples=pos_edge_index.size(1))

        F.nll_loss(model(pos_edge_index, neg_edge_index)[1][data.train_mask],
                   labels[data.train_mask]).backward()
        optimizer_attr.step()

    @torch.no_grad()
    def test_attr():
        model.eval()
        accs = []
        m = ['train_mask', 'val_mask', 'test_mask']
        i = 0
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):

            if (m[i] == 'train_mask'):
                x, pos_edge_index = data.x, data.train_pos_edge_index

                _edge_index, _ = remove_self_loops(pos_edge_index)
                pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                                   num_nodes=x.size(0))

                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                    num_neg_samples=pos_edge_index.size(1))
            else:
                pos_edge_index, neg_edge_index = [
                    index for _, index in data("{}_pos_edge_index".format(m[i].split("_")[0]),
                                               "{}_neg_edge_index".format(m[i].split("_")[0]))
                ]
            _, logits, _ = model(pos_edge_index, neg_edge_index)

            pred = logits[mask].max(1)[1]

            macro = f1_score((data.y[mask]).cpu().numpy(), pred.cpu().numpy(), average='macro')
            accs.append(macro)

            i += 1
        return accs

    best_val_acc = test_acc = 0
    for epoch in range(1, int(args.finetune_epochs) + 1):
        train_attr()
        train_acc, val_acc, tmp_test_acc = test_attr()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            res[m][seed][lam]['adversary'] = {'val': best_val_acc, 'test': test_acc}
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, tmp_test_acc))

    print(res)


if __name__ == '__main__':
    main()

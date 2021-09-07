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
from main import main

print(os.getcwd())

def evaluate():
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
    std_out = sys.stdout
    # GAL paper loops the following parameters. For our checking we will use first seed=100 lam=1.25 and m=GCNConv
    for dataset_var in ['cora', 'citeseer', 'pubmed']:
        for seed in [100, 200, 300, 400, 500]:
            for lam in [0, 1.25, 1.75]:
                for m in ['GINConv', 'GCNConv', 'GATConv', 'ChebConv']:
                    if args.SINGLE:
                        single_in_filename = "WITH_SINGLE"
                    else:
                        single_in_filename = "NOT_SINGLE"
                    output_filename = f"dataset_{dataset_var}.lam_{str(lam)}.seed_{str(seed)}." \
                                      f"GALconv_{m}_{single_in_filename}.txt"
                    sys.stdout = open(output_filename, 'w')
                    std_out.write('starting with ' + output_filename + ' run.\n')
                    args.seed = seed
                    args.lam = lam
                    args.GAL_gnn_type = m
                    args.dataset = dataset_var
                    main(args=args)
                    sys.stdout.close()
                    std_out.write('ending with ' + output_filename + ' run.\n')



if __name__ == '__main__':
    evaluate()

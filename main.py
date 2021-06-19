from our_models import *
from argparse import ArgumentParser
from torch_geometric.datasets import Planetoid
import os
from helper import *


def main():
    parser = ArgumentParser()
    parser.add_argument("-num_relations", dest="num_relations", default=5, required=False)
    parser.add_argument("-num_weights", dest="num_weights", default=2, required=False)
    parser.add_argument("-num_layers", dest="num_layers", default=2, required=False)
    parser.add_argument("-GAL_gnn_type", dest="GAL_gnn_type", default='GCNConv', required=False)
    parser.add_argument("-gpu", dest="num_layers", default=2, required=False)
    parser.add_argument("-dataset", dest="dataset", default='pubmed', required=False)
    args = parser.parse_args()

    if args.dataset == 'pubmed':
        dataset_path = os.path.join(getGitPath(), 'datasets')
        dataset = Planetoid(dataset_path, args.dataset)
        num_nodes = dataset.data.num_nodes
        num_edges = dataset.data.num_edges
        edges = dataset.data.edge_index
    num_hops = 2
    embed_dim = num_nodes
    # encoder = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2)
    decoder = SharedBilinearDecoder(args.num_relations, args.num_weights, embed_dim)
    model = OurGAL(decoder, embed_dim, num_nodes, edges, args, encoder=None, hop=num_hops)  # passing
    # encoder=None makes the model use the default nn.Embedding encoder


if __name__ == '__main__':
    main()

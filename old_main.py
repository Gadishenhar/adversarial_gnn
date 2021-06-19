from our_models import *
from argparse import ArgumentParser
from torch_geometric.datasets import Planetoid
import os
from helper import *
from torch.utils.data import Dataset, DataLoader


def main():
    parser = ArgumentParser()
    parser.add_argument("-num_relations", dest="num_relations", default=5, required=False)
    parser.add_argument("-num_weights", dest="num_weights", default=2, required=False)
    parser.add_argument("-num_layers", dest="num_layers", default=2, required=False)
    parser.add_argument("-GAL_gnn_type", dest="GAL_gnn_type", default='GCNConv', required=False)
    parser.add_argument("-gpu", dest="num_layers", default=2, required=False)
    parser.add_argument("-dataset", dest="dataset", default='pubmed', required=False)
    parser.add_argument("-batch", dest="batch", default=256, required=False)
    parser.add_argument("-lr", dest="lr", default=0.01, required=False)
    args = parser.parse_args()

    # Load the data set:
    if args.dataset == 'pubmed':
        dataset_path = os.path.join(getGitPath(), 'datasets')
        dataset = Planetoid(dataset_path, args.dataset)
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

    # define the attack model:
    train_fairness_set =
    # embeddings = defense_model.encode(None).detach().squeeze(0)
    # attack_model = NhopClassifier(embed_dim, embeddings, edges)

if __name__ == '__main__':
    main()

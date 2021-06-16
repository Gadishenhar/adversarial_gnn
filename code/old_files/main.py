from classes.basic_classes import GNN_TYPE, DataSet
from classes.attack_class import AttackMode
import sys
from argparse import ArgumentParser
from torch.cuda import set_device
import torch
from dataset_functions.graph_dataset import GraphDataset
from dataset_functions.twitter_dataset import TwitterDataset
from GAN import GANTrainer
from GAL.models import GAL, SharedBilinearDecoder
from model_functions.graph_model import NodeModel


def main():

    #Define parser for the arguments:
    parser = ArgumentParser()
    parser.add_argument("--attMode", dest="attMode", default=AttackMode.NODE, type=AttackMode.from_string,
                        choices=list(AttackMode), required=False)
    parser.add_argument("--dataset", dest="dataset", default=DataSet.TWITTER, type=DataSet.from_string,
                        choices=list(DataSet), required=False)

    parser.add_argument('--singleGNN', dest="singleGNN", type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                        required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=2, type=int, required=False)
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)

    parser.add_argument("--attEpochs", dest="attEpochs", default=20, type=int, required=False)
    parser.add_argument("--lr", dest="lr", type=float, default=0.1, required=False)
    parser.add_argument("--l_inf", dest="l_inf", type=float, default=None, required=False)
    parser.add_argument('--targeted', dest="targeted", action='store_true', required=False)

    parser.add_argument("--distance", dest='distance', type=int, required=False)

    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)

    parser.add_argument('--gpu', dest="gpu", type=int, required=False)
    parser.add_argument('--GAL', dest="GAL", type=bool, default=False, required=False)
    parser.add_argument('--GAL_gnn_type', dest="GAL_gnn_type", type=str, default="GATConv", required=False)
    args = parser.parse_args()

    # define the device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define the data set:
    dataset = GraphDataset(DataSet.TWITTER, device)
    num_layers = 2
    num_of_twitter_nodes = 4972 + 2  # TODO - verify this
    num_initial_features = dataset.num_features
    num_final_features = dataset.num_classes
    edges = dataset.data.edge_index.to(device)
    hidden_dims = [32] * (num_layers - 1)
    all_channels = [num_initial_features] + hidden_dims + [num_final_features]

    # create GNNs and optimizers:
    att_model = NodeModel(gnn_type=args.attMode.getGNN_TYPES(args=args), num_layers=num_layers, dataset=dataset, device=device, args=args)
    decoder = SharedBilinearDecoder(num_relations=5, num_weights=2, embed_dim=num_initial_features)
    def_model = GAL(decoder, num_ent=num_of_twitter_nodes, embed_dim=num_initial_features, edges=edges, args=args)
    att_adam_optimizer = torch.optim.Adam(att_model.parameters(), lr=0.0001)
    def_adam_optimizer = torch.optim.Adam(def_model.parameters(), lr=0.0001)

    # train the models:
    trainer = GANTrainer(att_model=att_model, att_optimzer=att_adam_optimizer, def_model=def_model,
                         def_optimzer=def_adam_optimizer)
    trainer.train(num_of_epochs=args.attEpochs)
    trainer.evaluate(args.attEpochs)

if __name__ == '__main__':
    main()

from our_models import *
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("-num_relations", dest="num_relations", default=5, required=False)
    parser.add_argument("-num_weights", dest="num_weights", default=2, required=False)
    parser.add_argument("-num_layers", dest="num_layers", default=2, required=False)
    args = parser.parse_args()

    num_nodes = 19717  # temporarily hard-coded, TODO - replace with initial number of nodes
    num_edges = 44324
    num_hops = 2
    embed_dim = num_nodes
    # encoder = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2)
    decoder = SharedBilinearDecoder(args.num_relations, args.num_weights, embed_dim)
    model = OurGAL(decoder, embed_dim, num_nodes, num_edges, args, encoder=None, hop=num_hops)  # passing
    # encoder=None makes the model use the default nn.Embedding encoder
    pass

if __name__ == '__main__':
    main()

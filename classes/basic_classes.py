from enum import Enum, auto
from torch import nn
from torch_geometric.nn import GCNConv
from classes.modified_gat import ModifiedGATConv
from classes.modified_gin import ModifiedGINConv
from classes.modified_sage import ModifiedSAGEConv
from GAL.models import GAL, SharedBilinearDecoder

num_of_twitter_nodes = 4972 + 2  # TODO - verify this


class Print(Enum):
    YES = auto()
    PARTLY = auto()
    NO = auto()


class DatasetType(Enum):
    CONTINUOUS = auto()
    DISCRETE = auto()


class DataSet(Enum):
    PUBMED = auto()
    CORA = auto()
    CITESEER = auto()
    TWITTER = auto()

    @staticmethod
    def from_string(s):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def get_type(self):
        if self is DataSet.PUBMED or self is DataSet.TWITTER:
            return DatasetType.CONTINUOUS
        elif self is DataSet.CORA or self is DataSet.CITESEER:
            return DatasetType.DISCRETE

    def string(self):
        if self is DataSet.PUBMED:
            return "PubMed"
        elif self is DataSet.CORA:
            return "Cora"
        elif self is DataSet.CITESEER:
            return "CiteSeer"
        elif self is DataSet.TWITTER:
            return "twitter"


class GNN_TYPE(Enum):
    GCN = auto()
    GAT = auto()
    SAGE = auto()
    GIN = auto()
    GAL = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim, args=None, edges=None):
        if self is GNN_TYPE.GCN:
            return GCNConv(in_channels=in_dim, out_channels=out_dim)
        elif self is GNN_TYPE.GAT:
            return ModifiedGATConv(in_channels=in_dim, out_channels=out_dim)
        elif self is GNN_TYPE.SAGE:
            return ModifiedSAGEConv(in_channels=in_dim, out_channels=out_dim)
        elif self is GNN_TYPE.GIN:
            sequential = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                       nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
            return ModifiedGINConv(sequential)
        elif self is GNN_TYPE.GAL:
            decoder = SharedBilinearDecoder(num_relations=5, num_weights=2, embed_dim=in_dim)
            return GAL(decoder, num_ent=num_of_twitter_nodes, embed_dim=in_dim, edges=edges, args=args)

    def string(self):
        if self is GNN_TYPE.GCN:
            return "GCN"
        elif self is GNN_TYPE.GAT:
            return "GAT"
        elif self is GNN_TYPE.SAGE:
            return "SAGE"
        elif self is GNN_TYPE.GIN:
            return "GIN"
        elif self is GNN_TYPE.GAL:
            return "GAL"

    @staticmethod
    def convertGNN_TYPEListToStringList(gnn_list):
        return [gnn.string() for gnn in gnn_list]

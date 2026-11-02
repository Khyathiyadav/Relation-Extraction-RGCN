
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class ImprovedRelationExtractionRGCN(nn.Module):

    def __init__(
        self,
        num_pos,
        num_relations,
        num_classes=19,
        pos_embedding_dim=32,
        hidden_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.pos_embedding = nn.Embedding(
            num_pos,
            pos_embedding_dim
        )

        input_dim = 768 + pos_embedding_dim

        self.rgcn1 = RGCNConv(
            input_dim,
            hidden_dim,
            num_relations=num_relations,
            num_bases=30
        )

        self.rgcn2 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations=num_relations,
            num_bases=30
        )

        self.dropout = nn.Dropout(dropout)

        pair_dim = hidden_dim * 4

        self.fc1 = nn.Linear(pair_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):

        bert_features = data.bert_x

        pos_ids = data.x.squeeze(-1).long()
        pos_features = self.pos_embedding(pos_ids)

        x = torch.cat(
            [bert_features, pos_features],
            dim=1
        )

        x = self.rgcn1(
            x,
            data.edge_index,
            data.edge_type
        )

        x = F.relu(x)
        x = self.dropout(x)

        x = self.rgcn2(
            x,
            data.edge_index,
            data.edge_type
        )

        x = F.relu(x)

        entity_pairs = []

        for i in range(data.num_graphs):

            graph_mask = data.batch == i

            e1_mask = data.entity_1_mask & graph_mask
            e2_mask = data.entity_2_mask & graph_mask

            e1 = x[e1_mask].mean(dim=0)
            e2 = x[e2_mask].mean(dim=0)

            difference = torch.abs(e1 - e2)
            interaction = e1 * e2

            pair = torch.cat(
                [e1, e2, difference, interaction],
                dim=0
            )

            entity_pairs.append(pair)

        pair_features = torch.stack(entity_pairs)

        out = self.fc1(pair_features)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

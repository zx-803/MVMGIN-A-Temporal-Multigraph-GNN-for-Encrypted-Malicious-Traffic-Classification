from torch_geometric.nn import MessagePassing
from basic_gnn import BasicGNN
from torch_geometric.utils import softmax
import torch

import torch.nn.functional as F
import torch.nn as nn
import importlib
import multi_view_dense_gnn
from multi_view_dense_gnn import MultiViewGNN


class BochnerTimeEncoder(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = torch.nn.Linear(1, hidden_dim)

    def forward(self, delta_t):

        return torch.sin(self.linear(delta_t.unsqueeze(-1)))


class CustomEdgeConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super().__init__(aggr='add')
        self.num_heads = num_heads
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * in_channels, 2 * in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * in_channels, hidden_channels)
        )
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(in_channels + hidden_channels, out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, edge_weights):

        x_multi = x.permute(1, 0, 2)

        outputs = []
        for h in range(self.num_heads):
            x = x_multi[h]
            aggr = self.propagate(
                edge_index,
                x=x,
                edge_attr=edge_attr,
                edge_weights=edge_weights[:, h]
            )
            combined = torch.cat([x, aggr], dim=1)
            head_output = self.phi(combined)
            outputs.append(head_output)

        return torch.stack(outputs, dim=1)

    def message(self, x_i, x_j, edge_attr, edge_weights):

        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)

        transformed = self.mlp(msg)  # [E, out]
        return transformed * edge_weights.unsqueeze(-1)  # [E, out]

    def _current_head(self):

        return getattr(self, '_processing_head', 0)


class MultiGraphLayer(torch.nn.Module):
    def __init__(self, in_channels=59, hidden_dim=128, n_heads=8, num_classes=4, num_layers=2,
                 num_nodes=None):
        super().__init__()
        self.n_heads = n_heads
        self.initial_node_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_classes = num_classes
        # self.gnn_convs = torch.nn.ModuleList([
        #     CustomEdgeConv(hidden_dim, hidden_dim, hidden_dim,n_heads)
        #     for _ in range(num_layers)
        # ])

        self.gnn_convs = torch.nn.ModuleList([
            MultiViewGNN(hidden_dim, hidden_dim, hidden_dim, num_views=n_heads, num_nodes=num_nodes)
            for _ in range(num_layers)
        ])

        self.edge_embed = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_dim, bias=True), torch.nn.ReLU())


        self.edge_weight_gen = torch.nn.Sequential(torch.nn.Linear(hidden_dim * 2, n_heads, bias=True),
                                                   torch.nn.Sigmoid())
        self.proj_layer = nn.Linear(n_heads * hidden_dim, num_classes * hidden_dim)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            kdim=hidden_dim,
            vdim=hidden_dim
        )



    def forward(self, x, edge_index, edge_attr):

        edge_attr = self.edge_embed(edge_attr)  # 原始流量表征

        h = x



        src_h = h[edge_index[0]]
        edge_weights = self.edge_weight_gen(torch.concat([edge_attr, src_h], dim=-1))  # [num_edges, n_view]\


        h = h.unsqueeze(1).expand(-1, self.n_heads, -1)  # [N, D] → [N, H, D]



        edge_index = edge_index[[1, 0], :]
        for conv in self.gnn_convs:

            h = h + conv(h, edge_index, edge_attr, edge_weights)

        return (h, edge_weights,
                edge_attr)


class GlobalLocalEncoder(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.local_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=False,
            kdim=2 * hidden_dim,
            vdim=2 * hidden_dim
        )

        self.global_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=False,
            kdim=hidden_dim,
            vdim=hidden_dim
        )

    def forward(self, edge_attr, global_feats, edge_index):

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]


        src_feats = global_feats[src_nodes]
        dst_feats = global_feats[dst_nodes]
        local_feats = torch.cat([src_feats, dst_feats], dim=-1)


        query = edge_attr.unsqueeze(0)
        key_local = local_feats.permute(1, 0, 2)

        local_out, _ = self.local_attn(
            query=query,
            key=key_local,
            value=key_local
        )  # 输出 [1, N, D]
        local_out = local_out.squeeze(0)

        pooled_global = global_feats.mean(dim=0)
        key_global = pooled_global.unsqueeze(1).repeat(1, edge_attr.size(0), 1)

        global_out, global_attn_weights = self.global_attn(
            query=query,
            key=key_global,
            value=key_global
        )
        global_out = global_out.squeeze(0)

        # 残差连接
        return torch.cat([edge_attr, local_out, global_out], dim=-1), pooled_global, global_attn_weights


class MVMGIN(torch.nn.Module):
    def __init__(self, num_nodes=93, in_dim=59, hidden_dim=128, n_heads=8, num_classes=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.memory_initial = torch.ones(num_nodes, hidden_dim)

        self.register_buffer('memory', self.memory_initial)


        self.snapshot_encoder = MultiGraphLayer(in_dim, hidden_dim, n_heads=n_heads, num_classes=num_classes)

        self.global_local = GlobalLocalEncoder(hidden_dim, hidden_dim)

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.time_encoder = BochnerTimeEncoder(hidden_dim)


        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )


        self.query = nn.Parameter(torch.randn(1, num_classes, hidden_dim))
        self.self_att = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )


    def update_memory(self, h, delta_t):

        N, num_classes, hidden_dim = h.shape


        time_encoding = self.time_encoder(delta_t)

        h_new = []
        for c in range(num_classes):

            h_c = h[:, c, :]

            h_c = h_c + time_encoding

            h_c_updated = self.gru_cell(h_c, self.memory[:N])  # [N, hidden_dim]
            h_new.append(h_c_updated)

        h_new = torch.stack(h_new, dim=1)

        attn_flat = h_new.reshape(N, -1)

        h_updated = self.update_nn(attn_flat)


        self.memory = h_updated.detach()

        return h_new, h_updated

    def forward(self, edge_index, edge_attr, delta_t=torch.tensor([60.0]).to('cuda')):

        x = self.memory

        h, edge_scores, edge_attr = self.snapshot_encoder(x, edge_index, edge_attr)
        h_before_update = h

        if delta_t is not None:
            h, _ = self.update_memory(h, delta_t)

        combined, pooled_global, global_attn_weights = self.global_local(
            edge_attr,
            h,
            edge_index
        )

        return self.classifier(combined), edge_scores, h_before_update, global_attn_weights

    def reset_memory(self):

        self.memory.copy_(self.memory_initial)

    def compute_loss(self, pred, labels, edge_scores, h, lambda_sp=0.00001, lambda_non_redundancy=0.1):

        loss1 = F.cross_entropy(pred, labels)
        non_redundancy_loss = 0
        for i in range(self.n_heads):
            for j in range(i + 1, self.n_heads):
                # h_i 和 h_j 的形状都是 [N, hidden_dim]
                h_i = h[:, i, :]
                h_j = h[:, j, :]

                cos_sim = F.cosine_similarity(h_i, h_j, dim=-1)  # 形状 [N]


                non_redundancy_loss += torch.mean(cos_sim ** 2)  # 对N个节点的相似度平方求平均


        num_pairs = self.n_heads * (self.n_heads - 1) / 2
        if num_pairs > 0:
            non_redundancy_loss /= num_pairs

        loss = loss1 + 0.001 * non_redundancy_loss

        return loss


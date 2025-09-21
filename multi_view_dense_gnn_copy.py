import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class MultiViewGNN(MessagePassing):
    def __init__(self, node_feat_dim, edge_dim, out_dim, num_views,num_nodes=None):
        super().__init__(aggr=None)
        self.node_feat_dim = node_feat_dim
        self.edge_dim = edge_dim
        self.num_views = num_views
        self.num_nodes =num_nodes


        self.psi = torch.nn.Linear(node_feat_dim + 2*node_feat_dim, out_dim)

        self.f = torch.nn.Linear(out_dim, out_dim)


        self.phi = torch.nn.Sequential(
                torch.nn.Linear(node_feat_dim + out_dim, out_dim),
                torch.nn.ReLU()
        )


    def forward(self, x, edge_index, edge_attr, edge_weights):


        x_multi = x.permute(1, 0, 2)

        head_outputs = []
        for h in range(self.num_views):
            view_weights = edge_weights[:, h].unsqueeze(-1)

            if not self.num_nodes:
                num_nodes =x.size(0)
            else:
                num_nodes = self.num_nodes
            aggr = self.propagate(
                edge_index,
                x=x_multi[h],
                edge_attr=edge_attr,
                edge_weights=view_weights,
                num_nodes=num_nodes

            )

            combined = torch.cat([x_multi[h], aggr], dim=1)
            head_output = self.phi(combined)
            head_outputs.append(head_output)

        return torch.stack(head_outputs, dim=1)

    def message(self, x_i, x_j, edge_attr, edge_weights):

        messages = torch.cat([edge_attr,x_i, x_j], dim=-1)


    def aggregate(self, inputs, edge_index, num_nodes=None):
        row, col = edge_index
        uv_pairs = torch.stack([row, col], dim=0).t()




        unique_uv, inverse,_ = torch.unique(uv_pairs,
                                          dim=0,
                                          return_inverse=True,
                                          return_counts=True)

        m_uv = scatter(inputs, inverse, dim=0, reduce='sum',
                       dim_size=unique_uv.shape[0])


        psi_out = torch.relu(self.psi(m_uv))




        v_indices = unique_uv[:, 1]

        aggr = scatter(psi_out, v_indices, dim=0,
                       dim_size=num_nodes, reduce='sum')

        return self.f(aggr)
    def aggregate_old(self, inputs, edge_index, num_nodes=None):

        row, col = edge_index
        uv_pairs = torch.stack([row, col], dim=0).t()

        v_indices = edge_index[1]


        unique_uv, inverse = torch.unique(uv_pairs, dim=0, return_inverse=True)
        m_uv = scatter(inputs, inverse, dim=0, reduce='sum')


        psi_out = torch.relu(self.psi(m_uv))


        v_indices = unique_uv[:, 1]
        aggr = scatter(psi_out, v_indices, dim=0,
                       dim_size=num_nodes, reduce='sum')
        return self.f(aggr)

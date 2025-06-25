import dgl
import dgl.function as fn
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import AvgPooling, GraphConv, MaxPooling
from dgl.ops import edge_softmax
from scipy.sparse import coo_matrix

from functions import edge_sparsemax
from torch import Tensor
from torch.nn import Parameter
from utils import get_batch_id, topk


# class WeightedGraphConv(GraphConv):

#     def forward(self, graph: DGLGraph, n_feat, e_feat=None):
#         if e_feat is None:
#             return super(WeightedGraphConv, self).forward(graph, n_feat)

#         with graph.local_scope():
#             if self.weight is not None:
#                 n_feat = torch.matmul(n_feat, self.weight)
#             src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
#             src_norm = src_norm.view(-1, 1)
#             dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
#             dst_norm = dst_norm.view(-1, 1)
#             n_feat = n_feat * src_norm
#             graph.ndata["h"] = n_feat
#             graph.edata["e"] = e_feat
#             graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))
#             n_feat = graph.ndata.pop("h")
#             n_feat = n_feat * dst_norm
#             if self.bias is not None:
#                 n_feat = n_feat + self.bias
#             if self._activation is not None:
#                 n_feat = self._activation(n_feat)
#             return n_feat

class WeightedGraphConv(GraphConv):
    def __init__(self, in_feats, out_feats):
        super(WeightedGraphConv, self).__init__(in_feats, out_feats, allow_zero_in_degree=True)
        # Attention parameters
        self.attention_W = nn.Parameter(torch.Tensor(in_feats, in_feats))
        self.attention_a = nn.Parameter(torch.Tensor(in_feats * 2, 1))
        # Initialize attention parameters
        nn.init.xavier_normal_(self.attention_W)
        nn.init.xavier_normal_(self.attention_a)

    def reset_parameters(self):
        # Initialize parent class parameters
        if self.weight is not None:
            nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # Initialize attention parameters if they exist
        if hasattr(self, 'attention_W'):
            nn.init.xavier_normal_(self.attention_W)
        if hasattr(self, 'attention_a'):
            nn.init.xavier_normal_(self.attention_a)

    def forward(self, graph: DGLGraph, n_feat, e_feat=None):
        with graph.local_scope():
            # Compute node attention scores
            h = torch.matmul(n_feat, self.attention_W)  # Shape: [num_nodes, in_feats]
            graph.ndata['h'] = h
            # Compute attention scores for source-destination pairs
            row, col = graph.all_edges()
            h_src = h[row]  # Source node features, shape: [num_edges, in_feats]
            h_dst = h[col]  # Destination node features, shape: [num_edges, in_feats]
            h_pairs = torch.cat([h_src, h_dst], dim=-1)  # Shape: [num_edges, in_feats * 2]
            attention_scores = F.leaky_relu(
                torch.matmul(h_pairs, self.attention_a).squeeze(-1), negative_slope=0.2
            )  # Shape: [num_edges]
            attention_scores = edge_softmax(graph, attention_scores)  # Normalize per destination node

            # Aggregate edge attention scores to nodes (max over incoming edges)
            graph.edata['attn'] = attention_scores
            graph.update_all(
                fn.copy_e('attn', 'm'),  # Copy edge attention scores
                fn.max('m', 'node_attn')  # Take max over incoming edges
            )
            node_attn_scores = graph.ndata.pop('node_attn')  # Shape: [num_nodes]
            node_attn_scores = node_attn_scores.unsqueeze(-1)  # Shape: [num_nodes, 1]

            # Scale node features by attention scores
            n_feat = n_feat * node_attn_scores  # Shape: [num_nodes, in_feats]

            # Existing WeightedGraphConv logic
            if self.weight is not None:
                n_feat = torch.matmul(n_feat, self.weight)
            src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5).view(-1, 1)
            dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).view(-1, 1)
            n_feat = n_feat * src_norm
            graph.ndata['h'] = n_feat
            graph.edata['e'] = e_feat if e_feat is not None else torch.ones(graph.num_edges(), device=n_feat.device)
            graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h'))
            n_feat = graph.ndata.pop('h') * dst_norm
            if self.bias is not None:
                n_feat = n_feat + self.bias
            if self._activation is not None:
                n_feat = self._activation(n_feat)
            return n_feat


class NodeInfoScoreLayer(nn.Module):
   
    def __init__(self, sym_norm: bool = True):
        super(NodeInfoScoreLayer, self).__init__()
        self.sym_norm = sym_norm

    def forward(self, graph: dgl.DGLGraph, feat: Tensor, e_feat: Tensor):
        with graph.local_scope():
            if self.sym_norm:
                src_norm = torch.pow(
                    graph.out_degrees().float().clamp(min=1), -0.5
                )
                src_norm = src_norm.view(-1, 1).to(feat.device)
                dst_norm = torch.pow(
                    graph.in_degrees().float().clamp(min=1), -0.5
                )
                dst_norm = dst_norm.view(-1, 1).to(feat.device)

                src_feat = feat * src_norm

                graph.ndata["h"] = src_feat
                graph.edata["e"] = e_feat
                graph = dgl.remove_self_loop(graph)
                graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))

                dst_feat = graph.ndata.pop("h") * dst_norm
                feat = feat - dst_feat
            else:
                dst_norm = 1.0 / graph.in_degrees().float().clamp(min=1)
                dst_norm = dst_norm.view(-1, 1)

                graph.ndata["h"] = feat
                graph.edata["e"] = e_feat
                graph = dgl.remove_self_loop(graph)
                graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))

                feat = feat - dst_norm * graph.ndata.pop("h")

            score = torch.sum(torch.abs(feat), dim=1)
            return score


class HGPSLPool(nn.Module):
    def __init__(
        self,
        in_feat: int,
        ratio=0.8,
        sample=True,
        sym_score_norm=True,
        sparse=True,
        sl=True,
        lamb=1.0,
        negative_slop=0.2,
        k_hop=3,
    ):
        super(HGPSLPool, self).__init__()
        self.in_feat = in_feat
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.k_hop = k_hop
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=sym_score_norm)
        
        # Edge attention parameter (for structure learning)
        if self.sl:
            self.att = Parameter(torch.Tensor(1, self.in_feat * 2))
        
        # Node attention parameters
        self.node_attn_W = Parameter(torch.Tensor(in_feat, in_feat))
        self.node_attn_a = Parameter(torch.Tensor(in_feat, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.sl:
            nn.init.xavier_normal_(self.att)
        nn.init.xavier_normal_(self.node_attn_W)
        nn.init.xavier_normal_(self.node_attn_a)

    def forward(self, graph: DGLGraph, feat: Tensor, e_feat=None):
        # Compute node attention scores
        h = torch.matmul(feat, self.node_attn_W)  # Shape: [num_nodes, in_feat]
        node_attn_scores = F.leaky_relu(
            torch.matmul(h, self.node_attn_a).squeeze(-1), negative_slope=0.2
        )  # Shape: [num_nodes]
        node_attn_scores = torch.sigmoid(node_attn_scores)  # Normalize to [0, 1]

        # Combine with existing node scores
        if e_feat is None:
            e_feat = torch.ones(
                (graph.num_edges(),), dtype=feat.dtype, device=feat.device
            )
        x_score = self.calc_info_score(graph, feat, e_feat)
        x_score = x_score * node_attn_scores  # Modulate scores with attention

        # Top-k pooling
        batch_num_nodes = graph.batch_num_nodes()
        perm, next_batch_num_nodes = topk(
            x_score, self.ratio, get_batch_id(batch_num_nodes), batch_num_nodes
        )
        feat = feat[perm]
        pool_graph = None
        if not self.sample or not self.sl:
            graph.edata["e"] = e_feat
            pool_graph = dgl.node_subgraph(graph, perm)
            e_feat = pool_graph.edata.pop("e")
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        if not self.sl:
            return pool_graph, feat, e_feat, perm, x_score

        # Structure Learning
        if self.sample:
            # Fast mode for large graphs
            row, col = graph.all_edges()
            num_nodes = graph.num_nodes()
            scipy_adj = coo_matrix(
                (
                    e_feat.detach().cpu().numpy(),
                    (row.detach().cpu().numpy(), col.detach().cpu().numpy()),
                ),
                shape=(num_nodes, num_nodes),
            )
            for _ in range(self.k_hop):
                two_hop = scipy_adj.dot(scipy_adj)
                two_hop = two_hop * (1e-5 / two_hop.max())
                scipy_adj = two_hop + scipy_adj
            # Remove duplicates by converting to a set of edges
            scipy_adj = scipy_adj.tocoo()
            edges = set(zip(scipy_adj.row, scipy_adj.col))
            row = torch.tensor([e[0] for e in edges], dtype=torch.long, device=graph.device)
            col = torch.tensor([e[1] for e in edges], dtype=torch.long, device=graph.device)
            e_feat = torch.tensor(
                scipy_adj.data[:len(edges)], dtype=torch.float, device=feat.device
            )

            # Filter edges for pooled nodes
            mask = perm.new_full((num_nodes,), -1)
            i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
            mask[perm] = i
            row, col = mask[row], mask[col]
            mask = (row >= 0) & (col >= 0)
            row, col, e_feat = row[mask], col[mask], e_feat[mask]

            # Add self-loops
            num_nodes = perm.size(0)  # Number of pooled nodes
            loop_index = torch.arange(
                0, num_nodes, dtype=row.dtype, device=row.device
            )
            loop_weight = torch.ones(
                (num_nodes,), dtype=e_feat.dtype, device=e_feat.device
            )  # Use 1.0 for self-loops
            e_feat = torch.cat([e_feat, loop_weight], dim=0)
            row = torch.cat([row, loop_index], dim=0)
            col = torch.cat([col, loop_index], dim=0)

            # Compute edge attention scores
            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + e_feat * self.lamb

            # Normalize edge weights
            sl_graph = dgl.graph((row, col), num_nodes=num_nodes, device=graph.device)
            sl_graph.set_batch_num_nodes(next_batch_num_nodes)
            if self.sparse:
                weights = edge_sparsemax(sl_graph, weights)
            else:
                weights = edge_softmax(sl_graph, weights)

            # Final graph
            mask = torch.abs(weights) > 0
            row, col, weights = row[mask], col[mask], weights[mask]
            pool_graph = dgl.graph((row, col), num_nodes=num_nodes, device=graph.device)
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)
            e_feat = weights

        else:
            # Complete graph mode (unchanged)
            batch_num_nodes = next_batch_num_nodes
            block_begin_idx = torch.cat(
                [
                    batch_num_nodes.new_zeros(1),
                    batch_num_nodes.cumsum(dim=0)[:-1],
                ],
                dim=0,
            )
            block_end_idx = batch_num_nodes.cumsum(dim=0)
            dense_adj = torch.zeros(
                (pool_graph.num_nodes(), pool_graph.num_nodes()),
                dtype=torch.float,
                device=feat.device,
            )
            for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
                dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.0
            row, col = torch.nonzero(dense_adj).t().contiguous()

            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            dense_adj[row, col] = weights

            pool_row, pool_col = pool_graph.all_edges()
            dense_adj[pool_row, pool_col] += self.lamb * e_feat
            weights = dense_adj[row, col]
            del dense_adj
            torch.cuda.empty_cache()

            complete_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(complete_graph, weights)
            else:
                weights = edge_softmax(complete_graph, weights)

            mask = torch.abs(weights) > 1e-9
            row, col, weights = row[mask], col[mask], weights[mask]
            e_feat = weights
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        return pool_graph, feat, e_feat, perm, x_score
    

class ConvPoolReadout(torch.nn.Module):

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        pool_ratio=0.8,
        sample: bool = False,
        sparse: bool = True,
        sl: bool = True,
        lamb: float = 1.0,
        pool: bool = True,
    ):
        super(ConvPoolReadout, self).__init__()
        self.use_pool = pool
        self.conv = WeightedGraphConv(in_feat, out_feat)
        if pool:
            self.pool = HGPSLPool(
                out_feat,
                ratio=pool_ratio,
                sparse=sparse,
                sample=sample,
                sl=sl,
                lamb=lamb,
            )
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=True)  # Added for non-pooling layers

    def forward(self, graph, feature, e_feat=None):
        out = F.relu(self.conv(graph, feature, e_feat))
        x_score = None
        perm = None
        if self.use_pool:
            graph, out, e_feat, perm, x_score = self.pool(graph, out, e_feat)

        else:
            # Compute scores even when not pooling
            x_score = self.calc_info_score(graph, out, e_feat)
            perm = torch.arange(graph.num_nodes(), device=graph.device)
        
        # readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        # return graph, out, e_feat, readout, x_score, perm
    
        readout = torch.cat(
            [self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1
        )
        return graph, out, e_feat, readout, x_score, perm

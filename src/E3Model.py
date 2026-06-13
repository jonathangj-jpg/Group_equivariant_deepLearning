import logging

import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace
from e3nn.util.test import assert_equivariant

class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out) -> None:
        super().__init__()

        tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([16, 256, tp.weight_numel], torch.relu)
        self.tp = tp
        self.irreps_out = self.tp.irreps_out

    def forward(self, node_features, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0 , reduce="mean")
        return node_features


class Network(torch.nn.Module):
    def __init__(self, lmax=1) -> None:
        super().__init__()

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        irreps = o3.Irreps("5x0e")

        self.convs = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        for _ in range(5):
            # Natural parity only (0e scalars, 1o vectors) with smooth activations:
            # silu on scalars, sigmoid on the gates
            gate = Gate(
                "40x0e",
                [torch.nn.functional.silu],  # scalar
                "20x0e",
                [torch.sigmoid],  # gates (scalars)
                "20x1o",  # gated tensors, num_irreps has to match with gates
            )
            self.convs.append(Convolution(irreps, self.irreps_sh, gate.irreps_in))
            self.gates.append(gate)
            irreps = gate.irreps_out

        # Final layer
        self.final = Convolution(irreps, self.irreps_sh, "0e")
        self.irreps_out = self.final.irreps_out

    def forward(self, data) -> torch.Tensor:

        edge_src, edge_dst = radius_graph(x=data.pos, r=4.5, batch=data.batch)
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_attr = o3.spherical_harmonics(l=self.irreps_sh, x=edge_vec, normalize=True, normalization="component")
        edge_length_embedded = (
            soft_one_hot_linspace(x=edge_vec.norm(dim=1), start=0.5, end=4.5, number=16, basis="smooth_finite", cutoff=True)
            * 16**0.5
        )

        # one hot encoding of batch of molecules
        x = (data.z[:, None] == torch.tensor([1,6,7,8,9], device=data.z.device)).to(data.pos.dtype)

        for conv, gate in zip(self.convs, self.gates):
            x = conv(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
            x = gate(x)
        x = self.final(x, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return scatter(x, data.batch, dim=0 , reduce="mean")

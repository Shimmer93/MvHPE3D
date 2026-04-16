"""Graph building blocks for joint-structured Stage 2 models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


SMPL_BODY23_JOINT_NAMES: tuple[str, ...] = (
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


SMPL_BODY23_BONES: tuple[tuple[int, int], ...] = (
    (2, 0),
    (2, 1),
    (2, 5),
    (0, 3),
    (1, 4),
    (5, 8),
    (3, 6),
    (4, 7),
    (6, 9),
    (7, 10),
    (8, 11),
    (8, 12),
    (8, 13),
    (11, 14),
    (12, 15),
    (13, 16),
    (15, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
)


def build_smpl_body23_graph() -> np.ndarray:
    """Return the 3-subset spatial graph used by CTR-style graph convolutions."""
    num_nodes = len(SMPL_BODY23_JOINT_NAMES)
    self_link = [(index, index) for index in range(num_nodes)]
    inward = [(dst, src) for src, dst in SMPL_BODY23_BONES]
    outward = list(SMPL_BODY23_BONES)
    return np.stack(
        (
            _edge_to_mat(self_link, num_nodes),
            _normalize_digraph(_edge_to_mat(inward, num_nodes)),
            _normalize_digraph(_edge_to_mat(outward, num_nodes)),
        )
    )


def _edge_to_mat(edges: list[tuple[int, int]] | tuple[tuple[int, int], ...], num_nodes: int) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edges:
        adjacency[dst, src] = 1.0
    return adjacency


def _normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=0)
    inv_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inv_degree[nonzero] = degree[nonzero] ** (-1)
    return adjacency @ np.diag(inv_degree)


def _init_conv(conv: nn.Conv2d) -> None:
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


def _init_bn(batch_norm: nn.BatchNorm2d, scale: float) -> None:
    nn.init.constant_(batch_norm.weight, scale)
    nn.init.constant_(batch_norm.bias, 0.0)


def _resolve_group_count(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, max(1, num_channels))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _build_norm(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_resolve_group_count(num_channels), num_channels)


class CTRGraphConv(nn.Module):
    """Channel-wise relational graph convolution adapted from CTR-GCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        relation_reduction: int = 8,
    ) -> None:
        super().__init__()
        relation_channels = 8 if in_channels in {3, 5, 6, 9} else max(8, in_channels // relation_reduction)
        self.query = nn.Conv2d(in_channels, relation_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, relation_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relation = nn.Conv2d(relation_channels, out_channels, kernel_size=1)
        self.activation = nn.Tanh()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _init_conv(module)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        query = self.query(x).mean(dim=-2)
        key = self.key(x).mean(dim=-2)
        value = self.value(x)
        relation = self.activation(query.unsqueeze(-1) - key.unsqueeze(-2))
        relation = self.relation(relation) * alpha + adjacency.unsqueeze(0).unsqueeze(0)
        return torch.einsum("ncuv,nctv->nctu", relation, value)


class CTRGraphBlock(nn.Module):
    """Residual multi-subset CTR-style graph block for inputs shaped [B, C, T, V]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        *,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        self.num_subsets = int(adjacency.shape[0])
        self.branches = nn.ModuleList(
            [CTRGraphConv(in_channels, out_channels) for _ in range(self.num_subsets)]
        )
        adjacency_tensor = torch.from_numpy(adjacency.astype(np.float32))
        if adaptive:
            self.adjacency = nn.Parameter(adjacency_tensor)
        else:
            self.register_buffer("adjacency", adjacency_tensor)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.output_norm = _build_norm(out_channels)
        if in_channels == out_channels:
            self.residual: nn.Module | None = None
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                _build_norm(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

        nn.init.constant_(self.output_norm.weight, 1e-6)
        nn.init.constant_(self.output_norm.bias, 0.0)
        if self.residual is not None:
            residual_conv = self.residual[0]
            residual_norm = self.residual[1]
            assert isinstance(residual_conv, nn.Conv2d)
            assert isinstance(residual_norm, nn.GroupNorm)
            _init_conv(residual_conv)
            nn.init.constant_(residual_norm.weight, 1.0)
            nn.init.constant_(residual_norm.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.adjacency.to(device=x.device, dtype=x.dtype)
        output = None
        for subset_index, branch in enumerate(self.branches):
            branch_output = branch(x, adjacency[subset_index], self.alpha)
            output = branch_output if output is None else output + branch_output
        assert output is not None
        output = self.output_norm(output)
        residual = x if self.residual is None else self.residual(x)
        return self.activation(output + residual)

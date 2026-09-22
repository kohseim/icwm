from __future__ import annotations
from typing import Any

def row_basis(torch: Any, matrix: Any) -> Any:
    matrix = matrix.float()
    if matrix.numel() == 0:
        return torch.empty((0, matrix.shape[-1]), dtype=torch.float32)
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(torch.float32).eps * singular.max()
    rank = int((singular > tolerance).sum())
    return vh[:rank].contiguous()

def novel_basis(torch: Any, candidate: Any, cumulative: Any) -> Any:
    basis = row_basis(torch, candidate)
    if cumulative.numel():
        basis = basis - (basis @ cumulative.transpose(0, 1)) @ cumulative
    return row_basis(torch, basis)

def sample_center(torch: Any, pooled: Any, mask: Any) -> Any:
    if int(mask.sum()) < 2:
        raise ValueError("sample centering requires at least two observed nodes")
    return pooled[mask].float().mean(dim=0)

def projected_component(
    torch: Any, pooled: Any, mask: Any, basis: Any
) -> tuple[Any, Any]:
    center = sample_center(torch, pooled, mask)
    centered = pooled.float() - center
    if not basis.numel():
        return torch.zeros_like(centered), center
    return (centered @ basis.transpose(0, 1)) @ basis, center

def complement_basis(torch: Any, graph_basis: Any, *, seed: int) -> Any:
    rank, width = graph_basis.shape
    if rank == 0:
        return torch.empty((0, width), dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate = torch.randn((rank, width), generator=generator, dtype=torch.float32)
    candidate -= (candidate @ graph_basis.transpose(0, 1)) @ graph_basis
    q, _r = torch.linalg.qr(candidate.transpose(0, 1), mode="reduced")
    basis = q.transpose(0, 1).contiguous()
    overlap = float((basis @ graph_basis.transpose(0, 1)).abs().max())
    if overlap > 1e-5:
        raise ValueError(f"random and graph bases overlap: {overlap}")
    return basis

def pairwise_coordinate_rms(torch: Any, pooled: Any, basis: Any, mask: Any) -> float:
    coordinates = pooled[mask].float() @ basis.transpose(0, 1)
    if coordinates.shape[0] < 2:
        return float("nan")
    left, right = torch.triu_indices(coordinates.shape[0], coordinates.shape[0], offset=1)
    return float((coordinates[left] - coordinates[right]).square().sum(dim=1).mean().sqrt())

def pairwise_contract_metrics(
    *, pre_pairwise: float, post_pairwise: float, clean_reference_pairwise: float
) -> dict[str, float]:

    return {
        "graph_pairwise_relative_to_current_ratio": (
            post_pairwise / max(pre_pairwise, 1e-8)
        ),
        "graph_pairwise_residual_ratio": (
            post_pairwise / max(clean_reference_pairwise, 1e-8)
        ),
        "random_graph_coordinate_preservation_error": (
            abs(post_pairwise - pre_pairwise) / max(clean_reference_pairwise, 1e-8)
        ),
    }

def quantization_refined_graph_ablation(
    torch: Any,
    hidden: Any,
    *,
    delta: Any,
    center: Any,
    graph_basis: Any,
    mask: Any,
    observed: list[int],
    by_node: list[list[int]],
    pool_hidden: Any,
    candidates: int = 4,
) -> tuple[Any, Any, float, float, int]:

    if candidates < 1:
        raise ValueError("quantization refinement requires at least one candidate")

    targets: dict[int, Any] = {}
    position_nodes: dict[int, int] = {}
    for node in observed:
        for position in by_node[node]:
            if position < hidden.shape[1]:
                targets[position] = hidden[:, position].float() + delta[node]
                position_nodes[position] = node
    if not targets:
        raise ValueError("graph ablation has no in-prefix token positions")

    modified = hidden.clone()
    best_values: dict[int, Any] = {}
    best_after = None
    best_post_pairwise = float("inf")
    initial_post_pairwise = float("nan")
    best_candidate = 0

    for candidate in range(1, candidates + 1):
        for position, target in targets.items():
            modified[:, position] = target.to(hidden.dtype)

        after, after_mask, _counts = pool_hidden(modified[0].float())
        after_mask = after_mask.to(mask.device)
        if not torch.equal(after_mask, mask):
            raise ValueError("post-patch node mask changed during quantization refinement")
        post_pairwise = pairwise_coordinate_rms(torch, after, graph_basis, mask)
        if candidate == 1:
            initial_post_pairwise = post_pairwise
        if post_pairwise < best_post_pairwise:
            best_post_pairwise = post_pairwise
            best_candidate = candidate
            best_after = after
            best_values = {
                position: modified[:, position].clone() for position in targets
            }

        residual = ((after - center) @ graph_basis.transpose(0, 1)) @ graph_basis
        for position, node in position_nodes.items():
            targets[position] = targets[position] - residual[node]

    for position, value in best_values.items():
        modified[:, position] = value
    if best_after is None:
        raise RuntimeError("quantization refinement did not produce a candidate")
    return (
        modified,
        best_after,
        initial_post_pairwise,
        best_post_pairwise,
        best_candidate,
    )

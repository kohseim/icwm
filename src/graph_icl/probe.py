from __future__ import annotations
import argparse
import hashlib
import math
import random
from typing import Any, Iterable
import numpy as np
from .metrics import binary_metrics, best_threshold

def walk_predictions(
    features: Any,
    mask: Any,
    adjacency: Any,
    projection: Any,
    alpha: Any,
    input_scale: float,
    indices: Iterable[int],
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    projection = projection.to(device)
    alpha = alpha.to(device)
    with torch.inference_mode():
        for index in indices:
            observed = np.flatnonzero(mask[index])
            if len(observed) < 2:
                continue
            tensor = torch.from_numpy(np.asarray(features[index], dtype=np.float32)).to(device)
            z = (tensor / input_scale) @ projection.transpose(0, 1)
            left, right = np.triu_indices(len(observed), 1)
            selected = torch.as_tensor(observed, device=device)
            local = z[selected]
            logits = alpha - (local[left] - local[right]).square().sum(dim=-1)
            rows.append(
                {
                    "row_index": int(index),
                    "observed": observed,
                    "labels": adjacency[np.ix_(observed, observed)][left, right].astype(np.int64),
                    "probabilities": torch.sigmoid(logits).cpu().numpy(),
                }
            )
    return rows

def train_probe(
    features: Any,
    mask: np.ndarray,
    adjacency: np.ndarray,
    train_indices: list[int],
    validation_indices: list[int],
    rank: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    hidden_size = features.shape[-1]
    observed_train = mask[train_indices]
    train_array = np.asarray(features[train_indices], dtype=np.float32)
    denominator = max(1, int(observed_train.sum()) * hidden_size)
    input_scale = float(np.sqrt(np.sum(train_array * train_array, dtype=np.float64) / denominator))
    input_scale = max(input_scale, 1e-6)
    del train_array
    projection = torch.nn.Parameter(torch.empty(rank, hidden_size, device=device))
    torch.nn.init.normal_(projection, mean=0.0, std=1.0 / math.sqrt(hidden_size * rank))
    alpha = torch.nn.Parameter(torch.tensor(1.0, device=device))
    optimizer = torch.optim.AdamW(
        [
            {"params": [projection], "weight_decay": args.weight_decay},
            {"params": [alpha], "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    positives = 0
    negatives = 0
    for index in train_indices:
        observed = np.flatnonzero(mask[index])
        left, right = np.triu_indices(len(observed), 1)
        labels = adjacency[np.ix_(observed, observed)][left, right]
        positives += int(labels.sum())
        negatives += int(len(labels) - labels.sum())
    positive_weight = float(negatives / max(1, positives))

    def epoch_loss(indices: list[int], training: bool) -> float:
        order = list(indices)
        if training:
            random.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            tensor = torch.from_numpy(np.asarray(features[batch_indices], dtype=np.float32)).to(device)
            batch_mask = torch.as_tensor(mask[batch_indices], dtype=torch.bool, device=device)
            z = (tensor / input_scale) @ projection.transpose(0, 1)
            distance = (z[:, :, None, :] - z[:, None, :, :]).square().sum(dim=-1)
            logits = alpha - distance
            left, right = torch.triu_indices(mask.shape[1], mask.shape[1], offset=1, device=device)
            valid = batch_mask[:, left] & batch_mask[:, right]
            labels = adjacency_tensor[left, right].expand(len(batch_indices), -1)
            element = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, left, right], labels, reduction="none")
            weights = torch.where(labels > 0.5, positive_weight, 1.0)
            per_walk = (element * weights * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            loss = per_walk.mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    best_state = None
    best_validation = math.inf
    stale = 0
    history = []
    for epoch in range(args.epochs):
        train_loss = epoch_loss(train_indices, True)
        with torch.no_grad():
            validation_loss = epoch_loss(validation_indices, False)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_state = (projection.detach().cpu().clone(), alpha.detach().cpu().clone(), epoch + 1)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    best_projection, best_alpha, best_epoch = best_state
    return {
        "projection": best_projection,
        "alpha": best_alpha,
        "input_scale": input_scale,
        "positive_weight": positive_weight,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "history": history,
    }

def random_projection(hidden_size: int, output_size: int, seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    standard_deviation = 1.0 / math.sqrt(hidden_size * output_size)
    matrix = rng.normal(
        0.0, standard_deviation, size=(output_size, hidden_size)
    ).astype(np.float32)
    digest = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    return matrix, digest

def fit_scalar_offset(distances: list[np.ndarray], labels: list[np.ndarray], positive_weight: float) -> float:
    beta = 1.0
    walks = len(distances)
    for _ in range(100):
        gradient = 0.0
        hessian = 0.0
        for distance, target in zip(distances, labels):
            probability = 1.0 / (1.0 + np.exp(np.clip(distance - beta, -80.0, 80.0)))
            weights = np.where(target > 0, positive_weight, 1.0) / max(1, len(target) * walks)
            gradient += float(np.sum(weights * (probability - target)))
            hessian += float(np.sum(weights * probability * (1.0 - probability)))
        step = gradient / max(hessian, 1e-12)
        beta -= step
        if abs(step) < 1e-8:
            break
    return float(beta)


def prediction_rows_from_distances(
    distances: list[np.ndarray], labels: list[np.ndarray], observed: list[np.ndarray], beta: float,
) -> list[dict[str, Any]]:
    return [
        {
            "row_index": index,
            "observed": obs,
            "labels": y,
            "probabilities": 1.0 / (1.0 + np.exp(np.clip(distance - beta, -80.0, 80.0))),
        }
        for index, (distance, y, obs) in enumerate(zip(distances, labels, observed))
    ]

def frozen_projection_distances(
    features: Any,
    mask: np.ndarray,
    adjacency: np.ndarray,
    indices: list[int],
    projection: Any,
    input_scale: float,
    device: Any,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    import torch

    projection = projection.to(device)
    distances: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected_indices = indices[start : start + batch_size]
            tensor = torch.from_numpy(np.asarray(features[selected_indices], dtype=np.float32)).to(device)
            projected = (tensor / input_scale) @ projection.transpose(0, 1)
            for offset, index in enumerate(selected_indices):
                observed = np.flatnonzero(mask[index])
                left, right = np.triu_indices(len(observed), 1)
                local = projected[offset, torch.as_tensor(observed, device=device)]
                distance = (local[left] - local[right]).square().sum(dim=-1)
                distances.append(distance.cpu().numpy())
                labels.append(adjacency[np.ix_(observed, observed)][left, right].astype(np.int64))
                observed_rows.append(observed)
    return distances, labels, observed_rows

def predict(features, mask, adjacency, fitted, indices, device):
    import torch
    if 'projection_sha256' in fitted:
        distances, labels, observed = frozen_projection_distances(
            features, mask, adjacency, list(indices), fitted['projection'],
            fitted['input_scale'], torch.device(device), fitted['batch_size'])
        return prediction_rows_from_distances(distances, labels, observed, float(fitted['alpha']))
    return walk_predictions(features, mask, adjacency, fitted['projection'], fitted['alpha'],
                            fitted['input_scale'], indices, torch.device(device))


def lock_threshold(features, mask, adjacency, fitted, validation, device):
    from .metrics import probe_metrics
    rows = predict(features, mask, adjacency, fitted, validation, device)
    fitted['threshold'] = best_threshold(np.concatenate([r['labels'] for r in rows]),
                                         np.concatenate([r['probabilities'] for r in rows]))
    fitted['validation'] = probe_metrics(rows, fitted['threshold'])
    return fitted


def fit_random(features, mask, adjacency, fitted, train, validation, rank, seed, device, batch_size=64):
    import torch
    matrix, matrix_hash = random_projection(features.shape[-1], rank, seed)
    projection = torch.from_numpy(matrix)
    distances, labels, _ = frozen_projection_distances(
        features, mask, adjacency, train, projection, fitted['input_scale'], torch.device(device), batch_size)
    beta = fit_scalar_offset(distances, labels, fitted['positive_weight'])
    result = {'projection': projection, 'alpha': torch.tensor(beta, dtype=torch.float64),
              'input_scale': fitted['input_scale'], 'projection_sha256': matrix_hash, 'batch_size': batch_size}
    return lock_threshold(features, mask, adjacency, result, validation, device)


def main():
    from types import SimpleNamespace
    from .extract import load_activation
    from .io import arguments, load_config, run_directory, selected_shots, save_checkpoint, load_checkpoint, write_json
    from .metrics import probe_metrics
    from .topology import adjacency_matrix, get_topology
    args = arguments('Fit adjacency probes and evaluate cross-shot transfer').parse_args()
    config = load_config(args.config)
    root, contract = run_directory(args, config)
    shots = selected_shots(args, config)
    layers = config['models'][args.model]['probe_layers']
    rank = config['topologies'][args.topology]['rank']
    adjacency = np.asarray(adjacency_matrix(get_topology(args.topology)), dtype=np.int8)
    train_args = SimpleNamespace(**config['probe'], device=args.device)
    for shot in shots:
        base = root / f'shot_{shot}'
        train_x, train_mask, train_meta = load_activation(base / 'activations/probe_train', contract)
        val_x, val_mask, val_meta = load_activation(base / 'activations/probe_validation', contract)
        if set(train_meta['families']) & set(val_meta['families']):
            raise ValueError('Probe split leakage')
        if train_meta['layers'] != layers or val_meta['layers'] != layers:
            raise ValueError('Probe activation layers differ from configuration')
        mask = np.concatenate([train_mask, val_mask])
        train = list(range(len(train_x)))
        validation = list(range(len(train_x), len(mask)))
        for axis, layer in enumerate(layers):
            path = base / 'probes' / f'layer_{layer}.pt'
            if path.exists():
                load_checkpoint(path, contract)
                continue
            features = np.concatenate([train_x[:, axis], val_x[:, axis]])
            learned = train_probe(features, mask, adjacency, train, validation, rank, train_args, config['probe']['seed'])
            lock_threshold(features, mask, adjacency, learned, validation, args.device)
            random_feature = fit_random(features, mask, adjacency, learned, train, validation, rank, config['probe']['seed'], args.device, config['probe']['batch_size'])
            save_checkpoint(path, {'contract': contract, 'shot': shot, 'layer': layer,
                                   'train_ids': train_meta['families'], 'validation_ids': val_meta['families'],
                                   'learned': learned, 'random_feature': random_feature})
            print(f'probe shot={shot} layer={layer}', flush=True)
            del features
    results = []
    for target in shots:
        features, mask, metadata = load_activation(root / f'shot_{target}' / 'activations/final_test', contract)
        if metadata['layers'] != layers:
            raise ValueError('Test activation layers differ from configuration')
        for source in shots:
            for axis, layer in enumerate(layers):
                checkpoint = load_checkpoint(root / f'shot_{source}' / 'probes' / f'layer_{layer}.pt', contract)
                if set(metadata['families']) & set(checkpoint['train_ids'] + checkpoint['validation_ids']):
                    raise ValueError('Test families overlap with probe training')
                for control in ('learned', 'random_feature'):
                    fitted = checkpoint[control]
                    rows = predict(features[:, axis], mask, adjacency, fitted, range(len(features)), args.device)
                    results.append({'source_shot': source, 'target_shot': target, 'layer': layer, 'control': control,
                                    **probe_metrics(rows, fitted['threshold'], config['probe']['bootstrap_samples'] if source == target else 0, config['probe']['seed'])})
    write_json(root / 'probe_metrics.json', results)


if __name__ == '__main__':
    main()

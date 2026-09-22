from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from .data import load_families, render_family
from .extract import collect
from .intervention_math import (complement_basis, novel_basis, pairwise_contract_metrics,
                                pairwise_coordinate_rms, projected_component,
                                quantization_refined_graph_ablation, row_basis)
from .io import (arguments, digest, file_hash, load_checkpoint, load_config, read_json,
                 run_directory, save_checkpoint, selected_shots, stable_seed, write_json, write_jsonl)
from .metrics import classification, intervention_summary, probe_metrics
from .model import load_runtime
from .probe import lock_threshold, predict, train_probe
from .topology import adjacency_matrix, get_topology


def seed_for(*values):
    return stable_seed(*values) % (2 ** 31)


def erasure_transform(bases, control='graph', random_bases=None, clean_reference=None):
    if control not in ('graph', 'random', 'null'):
        raise ValueError('Unknown intervention control')

    def transform(layer, hidden, before, mask, by_node, pool):
        basis = bases[layer].to(hidden.device)
        component, center = projected_component(torch, before, mask, basis)
        pre = pairwise_coordinate_rms(torch, before, basis, mask)
        reference = pre if clean_reference is None else clean_reference[layer]
        observed = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        l2_error = 0.0
        candidate = 0
        if control == 'null' or not basis.numel():
            modified, after = hidden, before
            delta = torch.zeros_like(before)
        elif control == 'graph':
            delta = -component
            modified, after, _, _, candidate = quantization_refined_graph_ablation(
                torch, hidden, delta=delta, center=center, graph_basis=basis,
                mask=mask, observed=observed, by_node=by_node, pool_hidden=pool)
        else:
            random_basis = random_bases[layer].to(hidden.device)
            random_component, _ = projected_component(torch, before, mask, random_basis)
            norm = component.norm(dim=1)
            delta = -random_component * (norm / random_component.norm(dim=1).clamp_min(1e-8))[:, None]
            l2_error = float(((delta.norm(dim=1)[mask] - norm[mask]).abs() / norm[mask].clamp_min(1e-8)).max())
            modified = hidden.clone()
            for node in observed:
                for position in by_node[node]:
                    modified[:, position] = (modified[:, position].float() + delta[node]).to(hidden.dtype)
            after, after_mask, _ = pool(modified[0].float())
            if not torch.equal(mask, after_mask):
                raise ValueError('Mask changed after random erasure')
        post = pairwise_coordinate_rms(torch, after, basis, mask)
        metrics = pairwise_contract_metrics(pre_pairwise=pre, post_pairwise=post, clean_reference_pairwise=reference)
        metrics.update(rank=int(basis.shape[0]), programmed_delta_l2=float(delta[mask].norm()),
                       actual_delta_l2=float((after - before)[mask].norm()),
                       nodewise_l2_match_relative_error=l2_error, quantization_candidate=candidate)
        return modified, metrics

    return transform


def fit_round(runtime, families, train_count, shot, layers, rank, old, config, output, contract, round_index, device):
    path = output / f'round_{round_index:02d}.pt'
    if path.exists():
        return load_checkpoint(path, contract)
    work = output / f'round_{round_index:02d}'
    old_layers = old['layers'] if old else {}
    old_bases = {layer: old_layers[layer]['basis'] for layer in layers} if old else {}
    if old and all(item['stopped'] for item in old_layers.values()):
        payload = {'contract': contract, 'round': round_index, 'layers': old_layers}
        save_checkpoint(path, payload)
        return payload
    transform = erasure_transform(old_bases) if old else None
    x, mask = collect(runtime, families, shot, layers, work / 'activations', digest([contract, 'fit', round_index]), transform)
    adjacency = np.asarray(adjacency_matrix(get_topology(families[0]['topology'])), dtype=np.int8)
    train = list(range(train_count))
    validation = list(range(train_count, len(families)))
    optimization = {**config['probe'], 'batch_size': config['intervention']['batch_size']}
    args = SimpleNamespace(**optimization, device=device)
    payloads = {}
    for axis, layer in enumerate(layers):
        layer_path = work / f'layer_{layer}.pt'
        if layer_path.exists():
            payloads[layer] = load_checkpoint(layer_path, contract)['payload']
            continue
        previous = old_layers.get(layer, {'basis': torch.empty((0, x.shape[-1])), 'streak': 0, 'stopped': False, 'history': []})
        if previous['stopped']:
            payloads[layer] = previous
            continue
        result = train_probe(x[:, axis], mask, adjacency, train, validation, rank, args,
                             seed_for(config['intervention']['seed'], rank, shot, round_index, layer))
        rows = predict(x[:, axis], mask, adjacency, result, validation, device)
        validation_auroc = probe_metrics(rows, 0.5)['auroc']
        if validation_auroc is None:
            raise ValueError('Validation AUROC is undefined')
        below = validation_auroc <= config['intervention']['stop_auroc']
        streak = previous['streak'] + 1 if below else 0
        basis = previous['basis']
        if not below:
            candidate = row_basis(torch, result['projection'] / result['input_scale'])
            novel = novel_basis(torch, candidate, basis)
            if len(novel) != rank:
                raise ValueError(f'Expected {rank} novel directions; got {len(novel)}')
            basis = row_basis(torch, torch.cat([basis, novel]))
        record = {'round': round_index, 'validation_auroc_before_removal': validation_auroc,
                  'basis_added': not below, 'cumulative_rank': len(basis),
                  'best_epoch': result['best_epoch'], 'best_validation_loss': result['best_validation_loss']}
        payloads[layer] = {'basis': basis, 'streak': streak,
                           'stopped': streak >= config['intervention']['stop_patience'],
                           'history': previous['history'] + [record]}
        save_checkpoint(layer_path, {'contract': contract, 'payload': payloads[layer]})
        print(f'INLP shot={shot} round={round_index} layer={layer} rank={len(basis)}', flush=True)
    payload = {'contract': contract, 'round': round_index, 'layers': payloads}
    save_checkpoint(path, payload)
    del x, mask
    for name in ('activations.npy', 'mask.npy', 'metadata.json', 'progress.json'):
        (work / 'activations' / name).unlink(missing_ok=True)
    return payload


def evaluate(runtime, families, shot, checkpoint, config, output, contract):
    layers = list(checkpoint['layers'])
    bases = {layer: item['basis'] for layer, item in checkpoint['layers'].items()}
    rank = config['topologies'][next(key for key in config['topologies'] if get_topology(key).name == families[0]['topology'])]['rank']
    random_bases = {layer: complement_basis(torch, basis, seed=seed_for(config['intervention']['seed'], rank, shot, layer, 'random')) for layer, basis in bases.items()}
    rows = []
    for index, family in enumerate(families):
        path = output / 'cases' / f'{index:06d}.json'
        if path.exists():
            saved = read_json(path)
            if saved['contract'] != contract or saved['prompt_family_id'] != family['prompt_family_id']:
                raise ValueError('Intervention evaluation checkpoint mismatch')
            rows.append(saved)
            continue
        prompt = render_family(family, shot)
        clean_reference = {}

        def capture(layer, hidden, before, mask, by_node, pool):
            clean_reference[layer] = pairwise_coordinate_rms(torch, before, bases[layer].to(hidden.device), mask)
            return hidden, {}

        scores, _, _, _ = runtime.forward(prompt, layers, capture, score=True)
        clean = {'scores': scores, **classification(scores, family['answer'])}
        row = {'contract': contract, 'prompt_family_id': family['prompt_family_id'],
               'answer': family['answer'], 'sequence_length': family['sequence_length'],
               'shot': shot, 'round': checkpoint['round'], 'clean': clean, 'null': clean}
        for control in ('graph', 'random'):
            transform = erasure_transform(bases, control, random_bases, clean_reference)
            scores, _, _, diagnostics = runtime.forward(prompt, layers, transform, score=True)
            for layer, info in diagnostics.items():
                if not info['rank']:
                    continue
                if control == 'graph' and info['graph_pairwise_residual_ratio'] > 0.03:
                    raise ValueError(f'Graph erasure residual exceeds 0.03 at layer {layer}')
                if control == 'random' and (info['random_graph_coordinate_preservation_error'] > 0.02 or info['nodewise_l2_match_relative_error'] > 1e-4):
                    raise ValueError(f'Random erasure control failed at layer {layer}')
            row[control] = {'scores': scores, **classification(scores, family['answer']), 'layers': diagnostics}
        write_json(path, row)
        rows.append(row)
        print(f'intervention shot={shot} round={checkpoint["round"]} family={index + 1}/{len(families)}', flush=True)
    write_jsonl(output / 'predictions.jsonl', rows)
    summary = intervention_summary(rows, config['intervention']['bootstrap_samples'], config['intervention']['seed'])
    write_json(output / 'summary.json', summary)
    return rows


def audit(runtime, families, counts, shot, checkpoint, config, output, contract, device):
    layers = list(checkpoint['layers'])
    bases = {layer: item['basis'] for layer, item in checkpoint['layers'].items()}
    rank = config['topologies'][next(key for key in config['topologies'] if get_topology(key).name == families[0]['topology'])]['rank']
    result_path = output / 'audit.json'
    if result_path.exists():
        if read_json(result_path)['contract'] != contract:
            raise ValueError('Audit contract mismatch')
        return
    x, mask = collect(runtime, families, shot, layers, output / 'audit_activations', contract, erasure_transform(bases))
    train = list(range(counts[0]))
    validation = list(range(counts[0], counts[0] + counts[1]))
    test = list(range(counts[0] + counts[1], len(families)))
    adjacency = np.asarray(adjacency_matrix(get_topology(families[0]['topology'])), dtype=np.int8)
    args = SimpleNamespace(**{**config['probe'], 'batch_size': config['intervention']['batch_size']}, device=device)
    results = {}
    for axis, layer in enumerate(layers):
        path = output / 'audit_probes' / f'layer_{layer}.pt'
        if path.exists():
            fitted = load_checkpoint(path, contract)['probe']
        else:
            fitted = train_probe(x[:, axis], mask, adjacency, train, validation, rank, args,
                                 seed_for(config['intervention']['seed'], rank, shot, layer, 'audit'))
            lock_threshold(x[:, axis], mask, adjacency, fitted, validation, device)
            save_checkpoint(path, {'contract': contract, 'probe': fitted})
        results[layer] = {'erased_rank': len(bases[layer]), 'threshold': fitted['threshold'],
                          'validation': fitted['validation'],
                          'test': probe_metrics(predict(x[:, axis], mask, adjacency, fitted, test, device), fitted['threshold'])}
    write_json(result_path, {'contract': contract, 'layers': results})
    del x, mask
    for name in ('activations.npy', 'mask.npy', 'metadata.json', 'progress.json'):
        (output / 'audit_activations' / name).unlink(missing_ok=True)


def main():
    args = arguments('Fit and evaluate iterative graph-subspace erasure').parse_args()
    config = load_config(args.config)
    root, contract = run_directory(args, config)
    shots = selected_shots(args, config, intervention=True)
    layers = config['intervention']['layers']
    if layers == 'all':
        layers = list(range(1, config['models'][args.model]['layers'] + 1))
    rank = config['topologies'][args.topology]['rank']
    splits = {split: load_families(args.data, config, args.topology, split) for split in config['splits']}
    combined = sum(splits.values(), [])
    if len({f['prompt_family_id'] for f in combined}) != len(combined):
        raise ValueError('Intervention split leakage')
    train_val = splits['probe_train'] + splits['probe_validation']
    audit_families = train_val + splits['final_test']
    counts = [len(splits['probe_train']), len(splits['probe_validation'])]
    runtime = load_runtime(config, args.model, args.device)
    for shot in shots:
        output = root / f'shot_{shot}' / 'intervention'
        fit_contract = digest([contract, 'INLP', shot])
        previous = None
        for round_index in range(1, config['intervention']['rounds'] + 1):
            previous = fit_round(runtime, train_val, counts[0], shot, layers, rank, previous, config,
                                 output, fit_contract, round_index, args.device)
            if round_index in config['intervention']['snapshots']:
                snapshot = output / f'snapshot_{round_index:02d}'
                snapshot_contract = digest([contract, shot, round_index, file_hash(output / f'round_{round_index:02d}.pt')])
                evaluate(runtime, splits['final_test'], shot, previous, config, snapshot, snapshot_contract)
                audit(runtime, audit_families, counts, shot, previous, config, snapshot, snapshot_contract, args.device)


if __name__ == '__main__':
    main()

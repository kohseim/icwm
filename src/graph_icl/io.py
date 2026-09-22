from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, values)).encode()).digest()[:8], 'big')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb' if binary else 'w', dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False))


def write_jsonl(path, rows):
    def write(f):
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_write(path, write)


def save_checkpoint(path, value):
    import torch
    atomic_write(path, lambda f: torch.save(value, f), binary=True)


def load_checkpoint(path, contract):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['contract'] != contract:
        raise ValueError(f'Checkpoint contract changed: {path}')
    return payload


def arguments(description, data=True):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=Path('configs/paper.json'))
    p.add_argument('--model', choices=('4b', '8b', '14b'), required=True)
    p.add_argument('--topology', required=True)
    if data:
        p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shots', type=int, nargs='+')
    p.add_argument('--device', default='cuda')
    return p


def load_config(path):
    from .topology import TOPOLOGIES
    c = read_json(path)
    if not set(c['topologies']).issubset(TOPOLOGIES):
        raise ValueError('Unsupported topology')
    if c['shots'] != sorted(set(c['shots'])) or not c['shots'] or c['shots'][0] != 0:
        raise ValueError('Shots must be unique, ordered and include zero')
    if not set(c['intervention']['shots']).issubset(c['shots']):
        raise ValueError('Intervention shots are absent from the data')
    if set(c['splits']) != {'probe_train', 'probe_validation', 'final_test'} or min(c['splits'].values()) < 1:
        raise ValueError('Invalid splits')
    if c['topologies'].get('lattice_3x5', {}).get('rule_examples', 3) != 3:
        raise ValueError('The 3 x 5 condition requires three rule examples')
    for name, spec in c['topologies'].items():
        offset = TOPOLOGIES[name].right_offset
        coordinates = 'row_column' if offset == (0, 2) else 'horizontal_vertical'
        if spec['offset'] != list(offset) or spec['rule_wrap'] or spec['coordinates'] != coordinates:
            raise ValueError(f'Unsupported coordinate or rule definition: {name}')
    if not c['intervention']['snapshots']:
        raise ValueError('At least one intervention snapshot is required')
    snapshots = c['intervention']['snapshots']
    if snapshots != sorted(set(snapshots)) or snapshots[0] < 1:
        raise ValueError('Snapshot rounds must be positive, ordered and unique')
    if c['intervention']['snapshots'][-1] != c['intervention']['rounds']:
        raise ValueError('The last snapshot must equal the final round')
    return c


def selected_shots(args, config, intervention=False):
    allowed = config['intervention']['shots'] if intervention else config['shots']
    shots = allowed if args.shots is None else args.shots
    if not shots or len(set(shots)) != len(shots) or not set(shots).issubset(allowed):
        raise ValueError('Requested shots are not in the configured shot grid')
    return sorted(shots)


def environment():
    versions = {}
    for package in ('torch', 'transformers', 'numpy', 'tokenizers', 'safetensors', 'huggingface-hub'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    source = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    revision = subprocess.run(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'versions': versions, 'source_sha256': digest(source), 'git_revision': revision.stdout.strip() or None}


def run_directory(args, config):
    if args.model not in config['models'] or args.topology not in config['topologies']:
        raise ValueError('Model/topology absent from configuration')
    root = args.out / args.model / args.topology
    manifest = read_json(args.data / 'manifest.json')
    env = environment()
    contract = {'config': config, 'dataset': manifest, 'model': args.model, 'topology': args.topology,
                'device': args.device, 'versions': env['versions'], 'source_sha256': env['source_sha256']}
    path = root / 'run.json'
    if path.exists():
        if read_json(path)['contract_hash'] != digest(contract):
            raise ValueError(f'Configuration, data, code or environment changed: {root}')
    else:
        write_json(path, {'contract_hash': digest(contract), **contract, 'git_revision': env['git_revision']})
    return root, digest(contract)

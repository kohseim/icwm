from pathlib import Path

import numpy as np

from .data import load_families, render_family
from .io import arguments, digest, file_hash, load_config, read_json, run_directory, selected_shots, write_json
from .model import load_runtime


def collect(runtime, families, shot, layers, output, contract, transform=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / 'metadata.json'
    metadata = {'contract': contract, 'families': [f['prompt_family_id'] for f in families],
                'shot': shot, 'layers': list(layers), 'state_ids': list(families[0]['state_to_word'])}
    if meta_path.exists():
        saved = read_json(meta_path)
        if saved['spec'] != metadata:
            raise ValueError('Activation checkpoint mismatch')
        if any(file_hash(output / name) != h for name, h in saved['files'].items()):
            raise ValueError('Activation checksum mismatch')
        return np.load(output / 'activations.npy', mmap_mode='r'), np.load(output / 'mask.npy')
    progress_path = output / 'progress.json'
    progress = read_json(progress_path) if progress_path.exists() else {'contract': digest(metadata), 'completed': 0}
    if progress['contract'] != digest(metadata):
        raise ValueError('Partial activation checkpoint mismatch')
    if not 0 <= progress['completed'] <= len(families):
        raise ValueError('Invalid activation resume position')
    shape = (len(families), len(layers), len(metadata['state_ids']), runtime.model.config.hidden_size)
    mode = 'r+' if progress['completed'] else 'w+'
    x = np.lib.format.open_memmap(output / 'activations.npy', mode=mode, dtype=np.float16, shape=shape)
    mask = np.lib.format.open_memmap(output / 'mask.npy', mode=mode, dtype=np.bool_, shape=(shape[0], shape[2]))
    if x.shape != shape or mask.shape != (shape[0], shape[2]):
        raise ValueError('Partial activation shape mismatch')
    for index in range(progress['completed'], len(families)):
        row = render_family(families[index], shot)
        if list(row['state_to_word']) != metadata['state_ids']:
            raise ValueError('Node ordering changed between families')
        _, captured, observed, _ = runtime.forward(row, layers, transform=transform)
        x[index] = np.stack([captured[layer] for layer in layers])
        mask[index] = observed
        if not np.isfinite(x[index]).all():
            raise ValueError('Non-finite activation')
        if (index + 1) % 10 == 0 or index + 1 == len(families):
            x.flush()
            mask.flush()
            write_json(progress_path, {'contract': digest(metadata), 'completed': index + 1})
            print(f'activation {index + 1}/{len(families)}', flush=True)
    x.flush()
    mask.flush()
    write_json(meta_path, {'spec': metadata, 'files': {name: file_hash(output / name) for name in ('activations.npy', 'mask.npy')}})
    return x, mask


def load_activation(path, contract):
    path = Path(path)
    metadata = read_json(path / 'metadata.json')
    if metadata['spec']['contract'] != contract:
        raise ValueError('Activation run contract mismatch')
    for name, h in metadata['files'].items():
        if file_hash(path / name) != h:
            raise ValueError(f'Activation checksum mismatch: {path / name}')
    return np.load(path / 'activations.npy', mmap_mode='r'), np.load(path / 'mask.npy'), metadata['spec']


def main():
    args = arguments('Extract pooled node representations').parse_args()
    config = load_config(args.config)
    root, contract = run_directory(args, config)
    shots = selected_shots(args, config)
    layers = config['models'][args.model]['probe_layers']
    runtime = load_runtime(config, args.model, args.device)
    for shot in shots:
        for split in config['splits']:
            families = load_families(args.data, config, args.topology, split)
            collect(runtime, families, shot, layers, root / f'shot_{shot}' / 'activations' / split, contract)


if __name__ == '__main__':
    main()

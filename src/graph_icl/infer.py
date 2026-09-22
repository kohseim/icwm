from .data import load_families, render_family
from .io import arguments, load_config, read_json, run_directory, selected_shots, write_json, write_jsonl
from .metrics import classification
from .model import load_runtime


def infer(runtime, families, shot, output, contract):
    rows = []
    for i, family in enumerate(families):
        path = output / 'cases' / f'{i:06d}.json'
        if path.exists():
            row = read_json(path)
            if row['contract'] != contract or row['prompt_family_id'] != family['prompt_family_id']:
                raise ValueError('Inference checkpoint mismatch')
        else:
            prompt = render_family(family, shot)
            scores, _, _, _ = runtime.forward(prompt, score=True)
            row = {'contract': contract, 'prompt_family_id': family['prompt_family_id'],
                   'answer': family['answer'], 'shot': shot, 'scores': scores,
                   **classification(scores, family['answer'])}
            write_json(path, row)
        rows.append(row)
    write_jsonl(output / 'predictions.jsonl', rows)
    write_json(output / 'summary.json', {'families': len(rows), 'accuracy': sum(r['correct'] for r in rows) / len(rows)})
    return rows


def main():
    args = arguments('Score every candidate node word').parse_args()
    config = load_config(args.config)
    root, contract = run_directory(args, config)
    shots = selected_shots(args, config)
    families = load_families(args.data, config, args.topology, 'final_test')
    runtime = load_runtime(config, args.model, args.device)
    for shot in shots:
        infer(runtime, families, shot, root / f'shot_{shot}' / 'inference', contract)
        print(f'{args.model}/{args.topology}/shot_{shot}: inference complete', flush=True)


if __name__ == '__main__':
    main()

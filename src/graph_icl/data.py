from __future__ import annotations
import argparse
import random
from typing import Any, Iterable, Mapping
from .topology import State, Topology, get_topology
from .io import stable_seed, write_json, write_jsonl, read_json, digest, file_hash
from pathlib import Path

FRUIT_WORDS: tuple[str, ...] = (
    "apple",
    "avocado",
    "banana",
    "berry",
    "cherry",
    "coconut",
    "cucumber",
    "date",
    "fig",
    "grape",
    "lemon",
    "lime",
    "mango",
    "olive",
    "orange",
    "peach",
    "pear",
    "pineapple",
    "plum",
    "prune",
    "pumpkin",
    "raspberry",
    "strawberry",
    "tomato",
    "citrus",
)

def lag_two_pair_present(states: Iterable[State], source: State, target: State) -> bool:
    pair = frozenset((source, target))
    sequence = tuple(states)
    return any(
        frozenset((left, right)) == pair
        for left, right in zip(sequence, sequence[2:])
    )

def _state_id(state: State) -> str:
    return "S_" + "_".join(str(value) for value in state)

def _mapping(topology: Topology, rng: random.Random) -> dict[State, str]:
    if topology.num_states > len(FRUIT_WORDS):
        raise ValueError(
            f"{topology.name} needs {topology.num_states} words; only "
            f"{len(FRUIT_WORDS)} fruit words are configured"
        )
    words = rng.sample(list(FRUIT_WORDS), topology.num_states)
    return dict(zip(topology.states, words))

def two_step_right_pairs(topology: Topology) -> tuple[tuple[State, State], ...]:

    if topology.dimension != 2:
        raise ValueError("two-step-right requires a two-dimensional topology")
    di, dj = topology.right_offset
    pairs = tuple(
        (source, (source[0] + di, source[1] + dj))
        for source in topology.states
        if (source[0] + di, source[1] + dj) in topology.state_set
    )
    if len(pairs) < 2:
        raise ValueError(f"topology {topology.name!r} has too few two-step-right pairs")
    return pairs

def render_two_step_right_prompt(
    sequence_words: Iterable[str],
    rule_examples: Iterable[Mapping[str, str]],
    query_source_word: str,
    *,
    answer_word: str | None = None,
    include_sequence: bool = True,
) -> str:

    lines: list[str] = []
    if include_sequence:
        lines.extend(
            (
                "Complete the pattern.",
                "",
                "[SEQUENCE]",
                ", ".join(str(word) for word in sequence_words),
                "",
            )
        )
    for index, example in enumerate(rule_examples):
        if index:
            lines.append("")
        lines.extend(
            (
                "[EXAMPLE]",
                f"Input: {example['input_word']} Output: {example['output_word']}",
            )
        )
    lines.extend(("", "[EXAMPLE]", f"Input: {query_source_word} Output:"))
    if answer_word is not None:
        lines.append(answer_word)
    return "\n".join(lines)

def _rule_example(
    mapping: Mapping[State, str], pair: tuple[State, State]
) -> dict[str, str]:
    source, target = pair
    return {
        "input_state": _state_id(source),
        "input_word": mapping[source],
        "output_state": _state_id(target),
        "output_word": mapping[target],
    }

def _awm_question(
    topology: Topology,
    mapping: Mapping[State, str],
    examples: Iterable[tuple[State, State]],
    query_pair: tuple[State, State],
) -> dict[str, Any]:
    source, target = query_pair
    extension_coordinates = topology.right_offset == (0, 2)
    coordinate_convention = (
        {
            "i_axis": "row",
            "j_axis": "column",
            "plot_x_axis": "j",
            "plot_y_axis": "-i",
            "positive_j_direction": "right",
        }
        if extension_coordinates
        else {
            "i_axis": "horizontal",
            "j_axis": "vertical",
            "positive_i_direction": "right",
        }
    )
    return {
        "rule_examples": [_rule_example(mapping, pair) for pair in examples],
        "query": {
            "source_state": _state_id(source),
            "source_word": mapping[source],
            "target_state": _state_id(target),
            "target_word": mapping[target],
            "rule": "two-step-right",
            "offset": list(topology.right_offset),
            "coordinate_convention": coordinate_convention,
        },
        "answer": mapping[target],
    }

def _sample_two_step_world(
    topology: Topology,
    *,
    sequence_length: int,
    rule_example_count: int,
    require_support_query: bool,
    allow_lag_two_copy: bool,
    rng: random.Random,
    max_attempts: int,
) -> dict[str, Any]:
    mapping = _mapping(topology, rng)
    valid_pairs = list(two_step_right_pairs(topology))
    required_observed_pairs = rule_example_count + 1 + int(require_support_query)
    if required_observed_pairs > len(valid_pairs):
        raise ValueError(
            f"{topology.name} needs {required_observed_pairs} distinct rule pairs; "
            f"only {len(valid_pairs)} exist"
        )
    for _attempt in range(max_attempts):
        current = rng.choice(topology.states)
        states = [current]
        for _ in range(sequence_length - 1):
            current = rng.choice(topology.neighbors(current))
            states.append(current)
        observed = set(states)
        observed_pairs = [
            pair for pair in valid_pairs if pair[0] in observed and pair[1] in observed
        ]
        anti_copy_pairs = [
            pair
            for pair in observed_pairs
            if not lag_two_pair_present(states, pair[0], pair[1])
        ]
        query_pairs = observed_pairs if allow_lag_two_copy else anti_copy_pairs
        needed_query_pairs = 2 if require_support_query else 1
        if (
            len(observed_pairs) >= required_observed_pairs
            and len(query_pairs) >= needed_query_pairs
        ):
            return {
                "mapping": mapping,
                "states": states,
                "observed": observed,
                "observed_pairs": observed_pairs,
                "anti_copy_pairs": anti_copy_pairs,
                "query_pairs": query_pairs,
                "required_observed_pairs": required_observed_pairs,
            }
    raise RuntimeError(
        f"failed to sample a strict AWM world for {topology.name} at "
        f"length {sequence_length}"
    )


def sample_episode(topology, length, examples, seed):
    rng = random.Random(seed)
    world = _sample_two_step_world(topology, sequence_length=length, rule_example_count=examples,
                                  require_support_query=False, allow_lag_two_copy=False, rng=rng, max_attempts=20_000)
    pair = rng.choice(world['query_pairs'])
    selected = rng.sample([p for p in world['observed_pairs'] if p != pair], examples)
    mapping = world['mapping']
    question = _awm_question(topology, mapping, selected, pair)
    return {**question, 'sequence_words': [mapping[s] for s in world['states']],
            'sequence_states': [_state_id(s) for s in world['states']],
            'sequence_length': length, 'rule_example_count': examples,
            'state_to_word': {_state_id(s): mapping[s] for s in topology.states},
            'answer_candidates': [mapping[s] for s in topology.states]}


def sample_family(config, topology_key, split, index):
    spec = config['topologies'][topology_key]
    topology = get_topology(topology_key)
    seed = spec['dataset_seed']
    family_seed = stable_seed(seed, 'two-step-right', topology.name, split, index)
    lower, upper = config['walk_length']
    length = random.Random(stable_seed(family_seed, 'length')).randint(lower, upper)
    final = sample_episode(topology, length, spec['rule_examples'], stable_seed(seed, split, topology.name, index, 'final'))
    final_pair = (final['query']['source_word'], final['query']['target_word'])
    supports = []
    for position in range(max(config['shots'])):
        length = random.Random(stable_seed(seed, split, topology.name, index, 'support-length', position)).randint(lower, upper)
        for attempt in range(20_000):
            demo = sample_episode(topology, length, spec['rule_examples'], stable_seed(seed, split, topology.name, index, 'support', position, attempt))
            exposed = {(e['input_word'], e['output_word']) for e in demo['rule_examples']}
            exposed.add((demo['query']['source_word'], demo['query']['target_word']))
            if final_pair not in exposed:
                supports.append(demo)
                break
        else:
            raise RuntimeError('Could not hold out the final lexical mapping')
    family_id = f'two-step-right:{topology.name}:{split}:{index:06d}'
    return {**final, 'prompt_family_id': family_id, 'topology': topology.name, 'split': split,
            'seed': family_seed, 'demonstrations': supports}


def render_family(family, shot):
    if not 0 <= shot <= len(family['demonstrations']):
        raise ValueError('Shot count exceeds the stored support bank')
    supports = family['demonstrations'][:shot]
    blocks = [render_two_step_right_prompt(d['sequence_words'], d['rule_examples'], d['query']['source_word'], answer_word=d['answer']) for d in supports]
    blocks.append(render_two_step_right_prompt(family['sequence_words'], family['rule_examples'], family['query']['source_word']))
    return {**family, 'demonstrations': supports, 'shot_count': shot, 'prompt': '\n\n\n'.join(blocks)}


def dataset_spec(config):
    return {k: config[k] for k in ('topologies', 'walk_length', 'shots', 'splits')}


def build_dataset(config, output):
    import json
    output = Path(output)
    manifest_path = output / 'manifest.json'
    contract = digest(dataset_spec(config))
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['contract'] != contract:
            raise ValueError('Existing dataset has different settings')
        for relative, h in manifest['files'].items():
            if file_hash(output / relative) != h:
                raise ValueError(f'Dataset checksum mismatch: {relative}')
        return
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / 'config.json'
    if spec_path.exists() and read_json(spec_path) != dataset_spec(config):
        raise ValueError('Partial dataset has different settings')
    write_json(spec_path, dataset_spec(config))
    files = {}
    for topology in config['topologies']:
        for split, count in config['splits'].items():
            relative = f'{topology}/{split}.jsonl'
            rows = (sample_family(config, topology, split, i) for i in range(count))
            write_jsonl(output / relative, rows)
            files[relative] = file_hash(output / relative)
            print(relative, count, flush=True)
    write_json(manifest_path, {'contract': contract, 'files': files})


def load_families(root, config, topology, split):
    import json
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    if manifest['contract'] != digest(dataset_spec(config)):
        raise ValueError('Dataset configuration mismatch')
    relative = f'{topology}/{split}.jsonl'
    path = root / relative
    if file_hash(path) != manifest['files'][relative]:
        raise ValueError(f'Dataset checksum mismatch: {relative}')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [r['prompt_family_id'] for r in rows]
    if len(rows) != config['splits'][split] or len(set(ids)) != len(ids):
        raise ValueError('Invalid family count or duplicate family IDs')
    for row in rows:
        if row['split'] != split or row['topology'] != get_topology(topology).name:
            raise ValueError('Dataset split/topology mismatch')
    return rows


def main():
    from .io import load_config
    p = argparse.ArgumentParser(description='Generate paired graph prompt families')
    p.add_argument('--config', type=Path, default=Path('configs/paper.json'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    build_dataset(load_config(args.config), args.out)


if __name__ == '__main__':
    main()

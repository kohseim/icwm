from __future__ import annotations
import copy
from collections import defaultdict
from typing import Any, Mapping, Sequence
import torch

def sequence_word_spans(sequence_words: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:

    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, word in enumerate(sequence_words):
        if index:
            parts.append(", ")
            cursor += 2
        start = cursor
        parts.append(str(word))
        cursor += len(str(word))
        spans.append((start, cursor))
    return "".join(parts), spans

def occurrence_token_indices(
    offsets: Sequence[Sequence[int]], spans: Sequence[tuple[int, int]]
) -> list[list[int]]:

    result: list[list[int]] = []
    for start, end in spans:
        indices = [
            index
            for index, offset in enumerate(offsets)
            if int(offset[0]) < end and int(offset[1]) > start
        ]
        if not indices:
            raise ValueError(f"no token overlaps sequence character span {(start, end)}")
        result.append(indices)
    return result

def pool_node_activations(
    hidden: Any,
    sequence_words: Sequence[str],
    occurrence_indices: Sequence[Sequence[int]],
    word_to_node: Mapping[str, int],
    num_nodes: int,
) -> tuple[Any, Any, Any]:

    import torch

    if len(sequence_words) != len(occurrence_indices):
        raise ValueError("word/occurrence length mismatch")
    by_node: dict[int, list[Any]] = defaultdict(list)
    for word, token_positions in zip(sequence_words, occurrence_indices):
        node = int(word_to_node[str(word)])
        by_node[node].append(hidden[list(token_positions)].float().mean(dim=0))
    pooled = torch.zeros((num_nodes, hidden.shape[-1]), dtype=torch.float32, device=hidden.device)
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=hidden.device)
    counts = torch.zeros(num_nodes, dtype=torch.int16, device=hidden.device)
    for node, occurrences in by_node.items():
        pooled[node] = torch.stack(occurrences).mean(dim=0)
        mask[node] = True
        counts[node] = len(occurrences)
    return pooled, mask, counts

def token_layout(tokenizer: Any, row: Mapping[str, Any], max_seq_len: int) -> dict[str, Any]:
    prompt = str(row["prompt"])
    candidates = [str(value) for value in row["answer_candidates"]]
    full_ids = [
        [int(value) for value in tokenizer(prompt + "\n" + candidate, add_special_tokens=False)["input_ids"]]
        for candidate in candidates
    ]
    common = min(len(values) for values in full_ids)
    for position in range(common):
        if len({values[position] for values in full_ids}) != 1:
            common = position
            break
    if common < 1 or any(common >= len(values) for values in full_ids):
        raise ValueError("candidate strings do not have a nonempty shared prompt prefix")
    if max(map(len, full_ids)) > max_seq_len:
        raise ValueError("prompt plus candidate exceeds --max-seq-len")

    sequence_words = [str(value) for value in row["sequence_words"]]
    _text, spans, _start = query_sequence_word_spans(
        prompt, sequence_words, int(row.get("shot_count", 0))
    )
    prompt_tokens = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    occurrence_positions = occurrence_token_indices(prompt_tokens["offset_mapping"], spans)

    prompt_ids = [int(value) for value in prompt_tokens["input_ids"]]
    touched = sorted({position for values in occurrence_positions for position in values})
    if not touched or max(touched) >= common:
        raise ValueError("final-query walk is not wholly inside the shared scoring prefix")
    for position in touched:
        if prompt_ids[position] != full_ids[0][position]:
            raise ValueError("walk tokenization changed under candidate concatenation")
    return {
        "candidates": candidates,
        "prefix_ids": full_ids[0][:common],
        "suffix_ids": [values[common:] for values in full_ids],
        "prompt_token_count": len(prompt_ids),
        "scoring_prefix_token_count": common,
        "occurrence_positions": occurrence_positions,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "prompt_offsets": [
            (int(offset[0]), int(offset[1]))
            for offset in prompt_tokens["offset_mapping"]
        ],
        "sequence_character_spans": [
            (int(span[0]), int(span[1])) for span in spans
        ],
    }

def node_position_map(row: Mapping[str, Any], layout: Mapping[str, Any]) -> tuple[list[list[int]], list[int], list[str]]:
    state_ids = list(row["state_to_word"])
    state_index = {state: index for index, state in enumerate(state_ids)}
    word_to_node = {
        str(word): state_index[str(state)] for state, word in row["state_to_word"].items()
    }
    by_node: list[list[int]] = [[] for _ in state_ids]
    counts = [0 for _ in state_ids]
    for word, positions in zip(row["sequence_words"], layout["occurrence_positions"]):
        node = word_to_node[str(word)]
        by_node[node].extend(int(value) for value in positions)
        counts[node] += 1
    return by_node, counts, state_ids

def pooled_from_hidden(torch: Any, hidden: Any, row: Mapping[str, Any], layout: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    state_ids = list(row["state_to_word"])
    state_index = {state: index for index, state in enumerate(state_ids)}
    word_to_node = {
        str(word): state_index[str(state)] for state, word in row["state_to_word"].items()
    }
    return pool_node_activations(
        hidden,
        [str(value) for value in row["sequence_words"]],
        layout["occurrence_positions"],
        word_to_node,
        len(state_ids),
    )

def walk_prefix_end(layout: Mapping[str, Any]) -> int:
    touched = sorted({
        int(position)
        for occurrence in layout["occurrence_positions"]
        for position in occurrence
    })
    if not touched:
        raise ValueError("query walk has no token positions")
    split = max(touched) + 1
    if split >= len(layout["prefix_ids"]):
        raise ValueError("instruction suffix is empty after query-walk split")
    if any(position >= split for position in touched):
        raise ValueError("query-walk split does not contain every walk token")
    return split

def staged_score_candidates(
    torch: Any,
    model: Any,
    layout: Mapping[str, Any],
    *,
    stage_a_handles: list[Any] | None = None,
) -> dict[str, float]:

    split = walk_prefix_end(layout)
    device = model.device
    stage_a_ids = torch.tensor(
        [layout["prefix_ids"][:split]], dtype=torch.long, device=device
    )
    stage_b_ids = torch.tensor(
        [layout["prefix_ids"][split:]], dtype=torch.long, device=device
    )
    handles = list(stage_a_handles or [])
    try:
        with torch.inference_mode():
            stage_a = model(
                input_ids=stage_a_ids,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()
    with torch.inference_mode():
        stage_b = model(
            input_ids=stage_b_ids,
            past_key_values=stage_a.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    cache = stage_b.past_key_values
    first_log_probs = stage_b.logits[0, -1].float().log_softmax(-1)
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for candidate, suffix in zip(layout["candidates"], layout["suffix_ids"]):
            score = float(first_log_probs[int(suffix[0])])
            if len(suffix) > 1:
                decode = torch.tensor([suffix[:-1]], dtype=torch.long, device=device)
                candidate_cache = copy.deepcopy(cache)
                output = model(
                    input_ids=decode,
                    past_key_values=candidate_cache,
                    use_cache=False,
                    return_dict=True,
                )
                targets = torch.tensor(suffix[1:], dtype=torch.long, device=device)
                score += float(
                    output.logits[0].float().log_softmax(-1)
                    .gather(1, targets[:, None]).sum()
                )
                del candidate_cache, output
            scores[str(candidate)] = score
    del stage_a, stage_b, cache
    return scores


def query_sequence_word_spans(prompt, sequence_words, demonstration_count):
    text, relative = sequence_word_spans(sequence_words)
    marker = '[SEQUENCE]\n'
    start = prompt.rfind(marker + text)
    if start < 0 or prompt.count(marker) != demonstration_count + 1:
        raise ValueError('Cannot locate the final query walk')
    start += len(marker)
    return text, [(start + a, start + b) for a, b in relative], start


class Runtime:
    def __init__(self, model, tokenizer, max_seq_len):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def forward(self, row, layers=(), transform=None, score=False):
        import numpy as np
        layout = token_layout(self.tokenizer, row, self.max_seq_len)
        by_node, _, _ = node_position_map(row, layout)
        captured = {}
        diagnostics = {}
        masks = {}
        handles = []
        for layer in layers:
            if not 1 <= layer <= len(self.model.model.layers):
                raise ValueError(f'Invalid block index: {layer}')

            def hook(module, inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                pool = lambda value: pooled_from_hidden(torch, value, row, layout)
                before, mask, _ = pool(hidden[0].float())
                if transform is not None:
                    hidden, info = transform(layer, hidden, before, mask, by_node, pool)
                    diagnostics[layer] = info
                after, after_mask, _ = pool(hidden[0].float())
                if not torch.equal(mask, after_mask):
                    raise ValueError('Node mask changed after intervention')
                captured[layer] = after.detach().cpu().numpy().astype(np.float16)
                masks[layer] = mask.cpu().numpy()
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            handles.append(self.model.model.layers[layer - 1].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                if score:
                    scores = staged_score_candidates(torch, self.model, layout, stage_a_handles=handles)
                else:
                    end = walk_prefix_end(layout)
                    ids = torch.tensor([layout['prefix_ids'][:end]], device=self.model.device)
                    self.model(input_ids=ids, use_cache=False, return_dict=True, logits_to_keep=1)
                    scores = None
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(layers):
            raise ValueError('Not every requested block hook ran')
        mask = next(iter(masks.values())) if masks else np.asarray([bool(p) for p in by_node])
        if any(not np.array_equal(mask, m) for m in masks.values()):
            raise ValueError('Node masks differ between blocks')
        return scores, captured, mask, diagnostics


def load_runtime(config, alias, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use a GPU node')
    spec = config['models'][alias]
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'],
                                               torch_dtype=getattr(torch, config['dtype']),
                                               attn_implementation='eager')
    if len(model.model.layers) != spec['layers'] or model.config.hidden_size != spec['hidden_size']:
        raise ValueError('Model architecture differs from the pinned configuration')
    model.to(device)
    return Runtime(model, tokenizer, config['max_seq_len'])

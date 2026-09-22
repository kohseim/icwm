from __future__ import annotations
import math
import numpy as np

def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = probabilities.astype(np.float64)
    predictions = probabilities > threshold
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if positives.any() and negatives.any():

        order = np.argsort(-probabilities, kind="mergesort")
        sorted_probabilities = probabilities[order]
        sorted_labels = labels[order]
        threshold_indices = np.r_[
            np.flatnonzero(np.diff(sorted_probabilities) != 0),
            len(sorted_probabilities) - 1,
        ]
        cumulative_tp = np.cumsum(sorted_labels)[threshold_indices]
        cumulative_fp = 1 + threshold_indices - cumulative_tp
        tpr = np.r_[0.0, cumulative_tp / positives.sum()]
        fpr = np.r_[0.0, cumulative_fp / negatives.sum()]
        auroc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5))
        precision_curve = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall_curve = cumulative_tp / positives.sum()
        average_precision = float(
            np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve)
        )
    else:
        auroc = math.nan
        average_precision = math.nan
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.unique(np.r_[0.0, np.linspace(0.01, 0.99, 99), probabilities, 1.0])
    order = np.argsort(probabilities, kind='stable')
    cumulative = np.r_[0, np.cumsum(labels[order])]
    cut = np.searchsorted(probabilities[order], candidates, side='right')
    tp = cumulative[-1] - cumulative[cut]
    predicted = len(labels) - cut
    precision = np.divide(tp, predicted, out=np.zeros(len(cut)), where=predicted != 0)
    recall = tp / cumulative[-1] if cumulative[-1] else np.zeros(len(cut))
    denominator = precision + recall
    scores = np.divide(2 * precision * recall, denominator, out=np.zeros(len(cut)), where=denominator != 0)
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def classification(scores, answer):
    values = np.array(list(scores.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite candidate scores')
    weights = np.exp(values - values.max())
    probability = float(weights[list(scores).index(answer)] / weights.sum())
    prediction = max(scores, key=lambda candidate: (scores[candidate], candidate))
    return {'prediction': prediction, 'correct': int(prediction == answer), 'correct_probability': probability}


def probe_metrics(rows, threshold, bootstrap_samples=0, seed=0):
    if not rows:
        raise ValueError('Empty probe evaluation')
    labels = np.concatenate([r['labels'] for r in rows])
    probabilities = np.concatenate([r['probabilities'] for r in rows])
    result = binary_metrics(labels, probabilities, threshold)
    if bootstrap_samples:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(bootstrap_samples):
            chosen = rng.integers(0, len(rows), len(rows))
            y = np.concatenate([rows[i]['labels'] for i in chosen])
            p = np.concatenate([rows[i]['probabilities'] for i in chosen])
            estimates.append(binary_metrics(y, p, threshold)['auroc'])
        finite = np.asarray(estimates)[np.isfinite(estimates)]
        result['auroc_ci95'] = np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in result.items()}


def intervention_summary(rows, bootstrap_samples, seed):
    clean = np.asarray([r['clean']['correct_probability'] for r in rows])
    graph = np.asarray([r['graph']['correct_probability'] for r in rows])
    random = np.asarray([r['random']['correct_probability'] for r in rows])
    if not len(rows) or np.any(clean <= 0):
        raise ValueError('Relative effects require positive clean probabilities')
    g = 100 * (clean - graph) / clean
    r = 100 * (clean - random) / clean
    values = np.stack([g, r, g - r], axis=1)
    lengths = np.asarray([row['sequence_length'] for row in rows])
    strata = [np.flatnonzero(lengths < 48), np.flatnonzero((lengths >= 48) & (lengths < 72)), np.flatnonzero(lengths >= 72)]
    strata = [s for s in strata if len(s)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([values[np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])].mean(axis=0)
                          for _ in range(bootstrap_samples)])
    result = {'families': len(rows)}
    for axis, name in enumerate(('graph_relative_degradation', 'random_relative_degradation', 'graph_minus_random')):
        result[name] = {'mean': float(values[:, axis].mean()),
                        'ci95': np.quantile(samples[:, axis], [0.025, 0.975]).tolist() if bootstrap_samples else None}
    for control in ('clean', 'graph', 'random', 'null'):
        result[control] = {'accuracy': float(np.mean([row[control]['correct'] for row in rows])),
                           'correct_probability': float(np.mean([row[control]['correct_probability'] for row in rows]))}
    return result

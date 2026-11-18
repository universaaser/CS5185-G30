"""Fit the frozen policy tree to development measurements with grouped leave-one-video-out validation.

Reads ``<run>/development.json`` produced by ``sweep.py``, searches shallow
constrained trees over the candidate space, selects the tree depth by grouped
leave-one-video-out validation, and freezes the final policy to ``rules.json``.
Fold details are written to ``<run>/calibration/folds.json``.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean

import features
from sweep import ROOT, RUN

OUTPUT = RUN / 'calibration'
OPTIONS = [('medium', 1), ('fast', 1), ('fast', 2)]
BASE = OPTIONS[0]


def measurements(row, option):
    return next(r for r in row['records']
                if (r['settings']['preset'], r['settings']['aq_mode']) == option)


def stats(pairs):
    return {
        'mean_ms_gain': mean(r['quality']['ms_ssim'] - b['quality']['ms_ssim'] for r, b in pairs),
        'worst_psnr_delta': min(r['quality']['psnr_y_db'] - b['quality']['psnr_y_db'] for r, b in pairs),
        'encode_ratio': sum(r['encode_s'] for r, _ in pairs) / sum(b['encode_s'] for _, b in pairs),
    }


def eligible(scores):
    return scores['worst_psnr_delta'] >= -.1 and scores['encode_ratio'] <= 1.25


def fit_tree(rows, depth):
    best_score = float('-inf')
    best_tree = None
    for option in OPTIONS:
        pairs = [(measurements(row, option), measurements(row, BASE)) for row in rows]
        scores = stats(pairs)
        gain = scores['mean_ms_gain'] * len(rows)
        if eligible(scores) and gain > best_score + 1e-14:
            best_score = gain
            best_tree = dict(preset=option[0], aq_mode=option[1], reason='Best feasible development leaf',
                             training_videos=len({r['video'] for r in rows}))
    if depth:
        for feature in range(len(features.NAMES)):
            values = sorted({row['x'][feature] for row in rows})
            for lower, upper in zip(values, values[1:]):
                threshold = (lower + upper) / 2
                left = [row for row in rows if row['x'][feature] <= threshold]
                right = [row for row in rows if row['x'][feature] > threshold]
                if min(len({r['video'] for r in part}) for part in (left, right)) < 2:
                    continue
                a, left_node = fit_tree(left, depth - 1)
                b, right_node = fit_tree(right, depth - 1)
                if a + b > best_score + 1e-14:
                    best_score = a + b
                    best_tree = dict(feature=feature, threshold=threshold, left=left_node, right=right_node)
    return best_score, best_tree


def policy(rows, depth):
    budgets = [r['x'][-1] for r in rows]
    return dict(tree=fit_tree(rows, depth)[1], budget_bpp_range=[min(budgets), max(budgets)])


def main():
    source = RUN / 'development.json'
    rows = json.loads(source.read_text(encoding='utf-8'))
    assert {r['video'][:2] for r in rows} == {'01', '02', '03', '04', '07', '08'}
    videos = sorted({r['video'] for r in rows})
    assert all(sorted(r['percent'] for r in rows if r['video'] == video) == [2, 5] for video in videos)
    trials = []
    for depth in range(3):
        pairs, folds = [], []
        for video in videos:
            fitted = policy([r for r in rows if r['video'] != video], depth)
            predictions = []
            for row in (r for r in rows if r['video'] == video):
                preset, aq, reason = features.rule_decision(fitted, row['x'])
                pair = (measurements(row, (preset, aq)), measurements(row, BASE))
                pairs.append(pair)
                predictions.append(dict(percent=row['percent'], preset=preset, aq=aq, reason=reason, **stats([pair])))
            folds.append(dict(held_out=video, predictions=predictions, **fitted))
        trials.append(dict(depth=depth, folds=folds, **stats(pairs)))
    candidates = [t for t in trials if eligible(t)]
    if candidates:
        selected = max(candidates, key=lambda t: (round(t['mean_ms_gain'], 14), -t['depth']))
        config = policy(rows, selected['depth'])
    else:
        selected = None
        config = policy(rows, 0)
        config['tree'] = dict(preset='medium', aq_mode=1, reason='No eligible depth in grouped validation')
    config.update(version='frozen', feature_names=features.NAMES, feature_version=features.FEATURE_VERSION,
                  selected_depth=selected['depth'] if selected else None,
                  validation={k: v for k, v in selected.items() if k != 'folds'} if selected else None,
                  development_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'folds.json').write_text(json.dumps(trials, indent=2), encoding='utf-8')
    (ROOT / 'rules.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config))


if __name__ == '__main__':
    main()

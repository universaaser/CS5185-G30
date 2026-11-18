"""Content analysis and frozen, interpretable x264 parameter selection.

``select(source, metadata, target_bytes)`` is the selector hook used by
``encode.py``.  The decision comes from the frozen decision tree in
``rules.json`` (overridable with ``CS5185_RULES``) guarded by the per-pixel
per-frame budget range seen during calibration: outside that range the fixed
baseline (medium/AQ1) is returned.  Runtime analysis failures (unreadable
video, decode anomaly, frame-count mismatch) also degrade to the fixed
baseline with the reason recorded in the job log, so one bad input never
breaks an encode batch.  Explicit configuration errors (rules incompatible
with the feature implementation, non-positive budget) still raise so invalid
measurements cannot happen silently.
"""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURE_VERSION = 'gray320'
NAMES = ['motion_mean', 'motion_p90', 'edge_mean', 'cut_fraction', 'budget_bpp']


def analyze(source):
    source = Path(source).resolve()
    started = perf_counter()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path(os.environ.get('CS5185_FEATURE_CACHE', ROOT / '.feature_cache'))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'{digest}-{FEATURE_VERSION}.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        return result, {'cache_hit': True, 'elapsed_s': perf_counter() - started}
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError('Cannot open video or determine frame rate')
    rows, previous = [], None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180), interpolation=cv2.INTER_AREA)
            motion = 0.0 if previous is None else float(np.abs(gray.astype(np.float32) - previous).mean() / 255)
            edge = float((cv2.Canny(gray, 100, 200) > 0).mean())
            rows.append({'time_s': len(rows) / fps, 'motion': motion, 'edge': edge, 'cut': int(motion > .20)})
            previous = gray.astype(np.float32)
    finally:
        cap.release()
    if not rows:
        raise ValueError('No frames decoded')
    windows = []
    for index in range(int(rows[-1]['time_s'] / .5) + 1):
        group = [r for r in rows if int(r['time_s'] / .5) == index]
        windows.append({'time_s': index * .5, **{k: float(np.mean([r[k] for r in group])) for k in ['motion', 'edge', 'cut']}})
    motion = [r['motion'] for r in rows[1:]] or [0.]
    result = {'version': FEATURE_VERSION, 'sha256': digest, 'frame_count': len(rows),
              'summary': {'motion_mean': float(np.mean(motion)), 'motion_p90': float(np.percentile(motion, 90)),
                          'edge_mean': float(np.mean([r['edge'] for r in rows])),
                          'cut_fraction': float(np.mean([r['cut'] for r in rows]))},
              'frames': rows, 'windows': windows}
    result['extraction_s'] = perf_counter() - started
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(path)
    return result, {'cache_hit': False, 'elapsed_s': perf_counter() - started}


def vector(summary, metadata, target_bytes):
    from fractions import Fraction
    return [summary[k] for k in NAMES[:-1]] + [8 * target_bytes / (metadata['duration_s'] * metadata['width'] * metadata['height'] * float(Fraction(metadata['fps'])))]


def rule_decision(config, values):
    node, trace = config['tree'], []
    bounds = config.get('budget_bpp_range')
    if bounds is not None:
        low, high = bounds
        budget = values[NAMES.index('budget_bpp')]
        covered = low <= budget <= high
        trace.append(f'budget_bpp={budget:.9g} {"inside" if covered else "outside"} training range [{low:.9g}, {high:.9g}]')
        if not covered:
            return 'medium', 1, trace[0] + ' (baseline fallback)'
    while 'feature' in node:
        index, threshold = node['feature'], node['threshold']
        value = values[index] if bounds is not None else float(np.float32(values[index]))
        left = value <= threshold
        trace.append(f'{NAMES[index]}={value:.6g} {"<=" if left else ">"} {threshold:.6g}')
        node = node['left' if left else 'right']
    return node['preset'], node['aq_mode'], '; '.join(trace + [node['reason']])


def select(source, metadata, target_bytes):
    started = perf_counter()
    try:
        analysis, timing = analyze(source)
        if analysis['frame_count'] != metadata['frames']:
            raise ValueError('Feature decoding did not preserve the full input frame count')
    except Exception as error:  # graceful degradation: never break the encode batch
        return {'preset': 'medium', 'aq_mode': 1, 'rule_version': 'baseline-fallback',
                'features': {}, 'analysis': {'cache_hit': False, 'elapsed_s': perf_counter() - started},
                'reason': f'Baseline fallback: feature analysis failed ({type(error).__name__}: {error})',
                'source_sha256': None}
    values = vector(analysis['summary'], metadata, target_bytes)
    if not all(np.isfinite(values)) or target_bytes <= 0:
        raise ValueError('Features and positive budget must be finite')
    config_path = Path(os.environ.get('CS5185_RULES', ROOT / 'rules.json'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['feature_version'] != FEATURE_VERSION or config['feature_names'] != NAMES:
        raise ValueError('Frozen rules do not match the feature implementation')
    preset, aq, reason = rule_decision(config, values)
    timing['selection_s'] = perf_counter() - started
    return {'preset': preset, 'aq_mode': aq, 'rule_version': config['version'],
            'features': dict(zip(NAMES, values)), 'analysis': timing,
            'reason': reason, 'source_sha256': analysis['sha256']}


def plot(source, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    analysis, _ = analyze(source)
    rows = analysis['windows']
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, key in zip(axes, ['motion', 'edge', 'cut']):
        ax.plot([r['time_s'] for r in rows], [r[key] for r in rows], marker='o')
        ax.set_ylabel(key)
        ax.set_ylim(bottom=0)
        if key == 'cut':
            ax.set_ylim(0, 1)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('Time (s), half-second windows')
    fig.suptitle(Path(source).name)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)

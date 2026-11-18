"""Resumable development sweep: measure the candidate grid on the six development videos.

Produces ``<run>/development.json`` (one record per task with content features and
per-candidate encoding/quality measurements) plus per-video feature curves and a
run ``environment.json``.  Every candidate encoding is cached on disk and reused
when the sweep is resumed.
"""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import encode
import features
import quality

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get('CS5185_RUN', ROOT / 'runs')).resolve()
DATA = Path(os.environ.get('CS5185_DATA', ROOT.parent / 'data/CS5185_Project_Video_Dataset')).resolve()
DEV = {1, 2, 3, 4, 7, 8}
CANDIDATES = [(p, a) for p in ['fast', 'medium', 'slow'] for a in [1, 2]]
BASE = CANDIDATES.index(('medium', 1))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def args():
    return SimpleNamespace(ffmpeg=quality.FFMPEG, ffprobe=quality.FFPROBE, threads=2, attempts=3)


def record_environment():
    if (RUN / 'environment.json').exists():
        return
    write(RUN / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(),
        'ffmpeg': encode.run([quality.FFMPEG, '-version']).splitlines()[0],
        'packages': {p: importlib.metadata.version(p) for p in
                     ['numpy', 'opencv-python', 'torch', 'pytorch-msssim', 'matplotlib']},
        'threads': 2,
        'cache_policy': 'SHA256 source + feature version; cold extraction and warm lookup reported separately'})


def experiment(source, percent, preset, aq, split):
    """Encode one (source, rate, fixed settings) candidate and measure quality."""
    directory = RUN / split / f'{preset}-aq{aq}' / str(percent)
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / f'{source.stem}.json'
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record['status'] == 'ok' and 'quality' in record:
            return record
    started = perf_counter()
    record = encode.encode_job(args(), source, percent, 'candidate',
                               lambda *unused: {'preset': preset, 'aq_mode': aq}, None, directory)
    write(record_path, record)
    if record['status'] != 'ok':
        raise RuntimeError(record.get('error'))
    metric_started = perf_counter()
    record['quality'] = quality.evaluate(source, Path(record['output']))
    record['eval_s'] = perf_counter() - metric_started
    record['validation_total_s'] = perf_counter() - started
    record['encode_s'] = sum(a.get('encode_s', 0) for a in record['attempts'])
    write(record_path, record)
    print(f'{split} {source.stem} {percent}% {preset}/AQ{aq}: '
          f'PSNR={record["quality"]["psnr_y_db"]:.3f}, MS={record["quality"]["ms_ssim"]:.6f}, '
          f'encode={record["encode_s"]:.2f}s', flush=True)
    return record


def sweep():
    rows = []
    for source in sorted(DATA.glob('*.mp4')):
        if int(source.name[:2]) not in DEV:
            continue
        analysis, timing = features.analyze(source)
        features.plot(source, RUN / f'{source.stem}_features.png')
        metadata, _, _ = encode.probe(quality.FFPROBE, source)
        for percent in [5, 2]:
            records = [experiment(source, percent, p, a, 'development') for p, a in CANDIDATES]
            rows.append({'video': source.name, 'percent': percent,
                         'x': features.vector(analysis['summary'], metadata, metadata['bytes'] * percent // 100),
                         'analysis_s': analysis['extraction_s'], 'records': records})
    write(RUN / 'development.json', rows)
    return rows


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    RUN.mkdir(exist_ok=True)
    record_environment()
    sweep()

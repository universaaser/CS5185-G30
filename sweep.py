"""Resumable development sweep: measure the candidate grid on the six development videos.

Produces ``<run>/development.json`` (one record per task with content features and
per-candidate encoding/quality measurements) plus per-video feature curves and a
run ``environment.json``.  Every candidate encoding is cached on disk and reused
when the sweep is resumed.
"""
import argparse
import hashlib
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


def file_digest(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def experiment(source, percent, preset, aq, split):
    """Encode one (source, rate, fixed settings) candidate and measure quality."""
    directory = RUN / split / f'{preset}-aq{aq}' / str(percent)
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / f'{source.stem}.json'
    options = args()
    output = directory / f'{source.stem}.mp4'
    identity = {
        'source_sha256': file_digest(source), 'percent': percent, 'preset': preset, 'aq': aq,
        'options': vars(options),
        'tool_versions': [encode.run([tool, '-version']).splitlines()[0]
                          for tool in (options.ffmpeg, options.ffprobe)],
        'implementation': [file_digest(Path(module.__file__)) for module in (encode, quality)],
        'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'pytorch-msssim')},
    }
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding='utf-8'))
        except (UnicodeError, json.JSONDecodeError):
            record = None
        if (isinstance(record, dict) and record.get('status') == 'ok' and 'quality' in record
                and record.get('identity') == identity and record.get('output') == str(output)
                and output.is_file() and record.get('output_sha256') == file_digest(output)):
            return record
    started = perf_counter()
    record = encode.encode_job(options, source, percent, 'candidate',
                               lambda *unused: {'preset': preset, 'aq_mode': aq}, None, directory)
    write(record_path, record)
    if record['status'] != 'ok':
        raise RuntimeError(record.get('error'))
    metric_started = perf_counter()
    record['quality'] = quality.evaluate(source, output, ffmpeg=options.ffmpeg, ffprobe=options.ffprobe)
    record['identity'] = identity
    record['output_sha256'] = file_digest(output)
    record['eval_s'] = perf_counter() - metric_started
    record['validation_total_s'] = perf_counter() - started
    record['encode_s'] = sum(a.get('encode_s', 0) for a in record['attempts'])
    write(record_path, record)
    print(f'{split} {source.stem} {percent}% {preset}/AQ{aq}: '
          f'PSNR={float(record["quality"]["psnr_y_db"]):.3f}, MS={record["quality"]["ms_ssim"]:.6f}, '
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

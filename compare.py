"""Full-course regression over 24 tasks: fixed baseline vs fixed fast vs content-adaptive.

Runs the frozen policy from ``rules.json`` on all 12 course videos at the 5% and
2% target sizes, against two fixed-setting references (medium/AQ1 baseline and
fast/AQ1), and writes ``<run>/regression/comparisons.csv`` and ``comparisons.json``.
All encodings are cached and reused when the comparison is resumed.
"""
import csv
import json

import encode
import features
from sweep import DATA, DEV, RUN, args, experiment, record_environment

REGRESSION = RUN / 'regression'
METHODS = ('baseline', 'fixed_fast', 'adaptive')


def sources():
    return sorted(DATA.glob('*.mp4'))


def split_of(source):
    number = int(source.name[:2])
    return 'development' if number in DEV else 'test' if number in {5, 6, 9, 10} else 'supplementary'


def main():
    record_environment()
    environment = json.loads((RUN / 'environment.json').read_text(encoding='utf-8'))
    if encode.run([args().ffmpeg, '-version']).splitlines()[0] != environment['ffmpeg']:
        raise ValueError('FFmpeg build differs from the recorded run environment')
    rows = []
    for source in sources():
        split = split_of(source)
        metadata, _, _ = encode.probe(args().ffprobe, source)
        for percent in [5, 2]:
            target_bytes = metadata['bytes'] * percent // 100
            decision = features.select(source, metadata, target_bytes)
            baseline = experiment(source, percent, 'medium', 1, split)
            fixed = experiment(source, percent, 'fast', 1, split)
            adaptive = baseline if (decision['preset'], decision['aq_mode']) == ('medium', 1) \
                else experiment(source, percent, decision['preset'], decision['aq_mode'], split)
            for method, record, reason, selection in (
                    ('baseline', baseline, 'Fixed', None),
                    ('fixed_fast', fixed, 'Fixed', None),
                    ('adaptive', adaptive, decision['reason'], decision['analysis'])):
                rows.append(dict(video=source.name, percent=percent, split=split, method=method,
                                 **record['settings'], psnr=record['quality']['psnr_y_db'],
                                 ms_ssim=record['quality']['ms_ssim'],
                                 utilization=record['result']['bytes'] / record['target_bytes'],
                                 bytes=record['result']['bytes'], budget=record['target_bytes'],
                                 encode_s=record['encode_s'],
                                 selection_s=selection.get('selection_s', 0.) if selection else 0.,
                                 cache_hit=selection.get('cache_hit', '') if selection else '',
                                 output=record['output'], attempts=len(record['attempts']),
                                 reason=reason, source_sha256=decision['source_sha256']))
            print(f'{split} {source.stem} {percent}% adaptive={decision["preset"]}/AQ{decision["aq_mode"]}', flush=True)
    if len(rows) != 24 * len(METHODS):
        raise ValueError('Expected 24 tasks and three methods')
    REGRESSION.mkdir(parents=True, exist_ok=True)
    with (REGRESSION / 'comparisons.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (REGRESSION / 'comparisons.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

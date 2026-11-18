"""Render regression tables, grouped summary statistics and same-timestamp frame comparisons.

Reads ``<run>/regression/comparisons.json`` produced by ``compare.py`` and writes
``summary.json`` plus a ``REPORT.md`` with the comparison tables and 1-second
frame strips (original / baseline / fixed fast / adaptive).
"""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sweep import DATA, RUN

REGRESSION = RUN / 'regression'
FIGURE_TASKS = [('01_FourPeople.mp4', 5), ('09_Parkrun.mp4', 5), ('11_Sintel.mp4', 2)]
METHOD_LABELS = (('baseline', 'Baseline'), ('fixed_fast', 'Fixed fast'), ('adaptive', 'Adaptive'))


def differences(group, method, reference):
    changes = {}
    for metric in ('psnr', 'ms_ssim'):
        delta = [task[method][metric] - task[reference][metric] for task in group]
        changes[metric] = dict(mean=mean(delta), worst=min(delta), best=max(delta),
                               better=sum(v > 0 for v in delta), worse=sum(v < 0 for v in delta),
                               equal=sum(v == 0 for v in delta))
    changes['encode_ratio'] = sum(t[method]['encode_s'] for t in group) / \
        sum(t[reference]['encode_s'] for t in group)
    return changes


def figures(tasks):
    for video, percent in FIGURE_TASKS:
        task = tasks[video, percent]
        paths = [DATA / video] + [Path(task[m]['output']) for m, _ in METHOD_LABELS]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for ax, path, label in zip(axes.flat, paths, ['Original'] + [label for _, label in METHOD_LABELS]):
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError(f'Cannot read frame 30 from {path}')
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(label)
            ax.axis('off')
        fig.suptitle(f'{video} | {percent}% | 1.000 s (frame 30)')
        fig.tight_layout()
        fig.savefig(REGRESSION / f'{Path(video).stem}-{percent}.png', dpi=160)
        plt.close(fig)


def main():
    rows = json.loads((REGRESSION / 'comparisons.json').read_text(encoding='utf-8'))
    config = json.loads((Path(__file__).resolve().parent / 'rules.json').read_text(encoding='utf-8'))
    tasks = defaultdict(dict)
    for row in rows:
        tasks[row['video'], row['percent']][row['method']] = row
    summary = {}
    for name in ['all', 'development', 'test', 'supplementary']:
        group = [t for t in tasks.values() if name == 'all' or t['adaptive']['split'] == name]
        summary[name] = {f'{a}_vs_{b}': differences(group, a, b) for a, b in
                         [('adaptive', 'baseline'), ('fixed_fast', 'baseline'), ('adaptive', 'fixed_fast')]}
    (REGRESSION / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    bounds = config['budget_bpp_range']
    lines = ['# 课程数据回归验证', '',
             '六个开发视频用于拟合并冻结策略，其余六个视频按保留测试集（05/06/09/10）与补充案例（11/12）报告。'
             '三种方法共用 FFmpeg、两线程、字节预算与完整性检查；相同的输出复用已有耗时和指标。', '',
             f"冻结策略：每像素每帧预算处于训练区间（约 {bounds[0]:.4f} 至 {bounds[1]:.4f}）时按冻结树决策，区间外统一采用 medium/AQ1 基线。实际决策使用冻结文件内的完整精度。", '',
             '| 集合 | 比较 | 平均 ΔPSNR (dB) | 最差 ΔPSNR | 改善/持平/退化 | 平均 ΔMS-SSIM | 最差 ΔMS-SSIM | 改善/持平/退化 | 编码耗时比 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, comparisons in summary.items():
        for key, change in comparisons.items():
            p, m = change['psnr'], change['ms_ssim']
            lines.append(f"| {name} | {key} | {p['mean']:+.5f} | {p['worst']:+.5f} | "
                         f"{p['better']}/{p['equal']}/{p['worse']} | {m['mean']:+.8f} | {m['worst']:+.8f} | "
                         f"{m['better']}/{m['equal']}/{m['worse']} | {change['encode_ratio']:.4f} |")
    lines += ['', '## 各任务自适应决策与结果', '',
              '| 视频 | 档位 | 决策 | ΔPSNR (dB) | ΔMS-SSIM | 预算利用率 | 基线/自适应编码(s) |',
              '|---|---:|---|---:|---:|---:|---:|']
    for (video, percent), t in sorted(tasks.items()):
        base, adaptive = t['baseline'], t['adaptive']
        lines.append(f"| {video} | {percent}% | {adaptive['preset']}/AQ{adaptive['aq_mode']} | "
                     f"{adaptive['psnr'] - base['psnr']:+.5f} | {adaptive['ms_ssim'] - base['ms_ssim']:+.8f} | "
                     f"{adaptive['utilization']:.2%} | {base['encode_s']:.2f}/{adaptive['encode_s']:.2f} |")
    lines += ['', '## 复查材料', '',
              '[三种方法的逐任务数据](comparisons.csv) 包含原始 PSNR、MS-SSIM、文件大小、预算、编码耗时、'
              '选参耗时、输出视频位置与决策原因。编码时间包含超预算重试；冷分析与暖缓存选参耗时由特征缓存键（原片 SHA256 加特征算法版本）区分。', '',
              '[分组统计](summary.json) · [冻结配置](../../rules.json)', '']
    figures(tasks)
    for video, percent in FIGURE_TASKS:
        name = f'{Path(video).stem}-{percent}'
        lines += [f'![{name}，1 秒处原片、基线、固定 fast 与自适应输出]({name}.png)', '']
    (REGRESSION / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Adaptive vs baseline:', summary['all']['adaptive_vs_baseline'])


if __name__ == '__main__':
    main()

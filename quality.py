"""Validation-only full-frame luma metrics, with timestamp checks.

Also serves as the example ``--evaluator quality:evaluate`` hook for
``encode.py``: it is called only on compliant, complete candidate MP4s.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from pytorch_msssim import ms_ssim
import encode

ROOT = Path(__file__).resolve().parent
FFMPEG = os.environ.get('CS5185_FFMPEG', str(ROOT.parent / 'ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe'))
FFPROBE = os.environ.get('CS5185_FFPROBE', str(Path(FFMPEG).with_name('ffprobe.exe')))
torch.set_num_threads(2)


def luma(path, width, height):
    # Decode the Y plane directly; do not convert limited-range Y to full-range RGB.
    result = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-map', '0:v:0',
        '-vf', 'extractplanes=y', '-pix_fmt', 'gray', '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1'], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width)


def evaluate(source, encoded):
    meta, duration, times = encode.probe(FFPROBE, Path(source))
    other, other_duration, other_times = encode.probe(FFPROBE, Path(encoded))
    if any(meta[k] != other[k] for k in ['width', 'height', 'frames', 'fps']) or abs(duration-other_duration) > .001:
        raise ValueError('Reference/output geometry or duration mismatch')
    if any(abs((a-times[0])-(b-other_times[0])) > .001 for a,b in zip(times,other_times)):
        raise ValueError('Reference/output timestamps mismatch')
    ref, out = (luma(p, meta['width'], meta['height']) for p in [source, encoded])
    if ref.shape != out.shape or len(ref) != meta['frames']:
        raise ValueError('Decoded frame count mismatch')
    squared_error, scores = 0., []
    with torch.inference_mode():
        for start in range(0, len(ref), 4):
            a = torch.from_numpy(ref[start:start+4].copy()).unsqueeze(1).float()
            b = torch.from_numpy(out[start:start+4].copy()).unsqueeze(1).float()
            squared_error += float(((a.double()-b.double())**2).sum())
            scores.extend(ms_ssim(a,b,data_range=255,size_average=False).tolist())
    mse = squared_error / ref.size
    return {'psnr_y_db': 10*math.log10(255**2/mse) if mse else 'inf',
            'ms_ssim': float(np.mean(scores)), 'frames': len(ref),
            'metric_config': {'implementation': 'pytorch-msssim 1.0.0', 'channel': 'decoded Y',
                              'range': 255, 'scales': 5, 'temporal': 'arithmetic mean per frame',
                              'psnr': 'global pixel/frame MSE', 'torch_threads': 2}}

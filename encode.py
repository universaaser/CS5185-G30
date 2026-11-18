"""Batch H.264 encoding for the CS5185 video dataset."""
import argparse
from fractions import Fraction
from functools import partial
import importlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter


BASELINE = {"preset": "medium", "aq_mode": 1}
PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower", "veryslow", "placebo"}


def run(command):
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def probe(ffprobe, path, packets=False):
    command = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_streams", "-show_format", "-show_frames",
               "-show_entries", "stream=codec_name,width,height,avg_frame_rate,duration:"
               "format=duration:frame=best_effort_timestamp_time"]
    if packets:
        command[-1] += ":packet=size"
        command += ["-show_packets"]
    data = json.loads(run([*command, "-of", "json", str(path)]))
    stream = data["streams"][0]
    entries = data.get("packets_and_frames", data.get("frames", []))
    timestamps = [Fraction(item["best_effort_timestamp_time"]) for item in entries
                  if item.get("type", "frame") == "frame"]
    if not timestamps:
        raise ValueError(f"No timestamped video frames in {path}")
    duration = Fraction(stream.get("duration") or data["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Invalid video duration in {path}")
    metadata = {"bytes": path.stat().st_size, "duration_s": float(duration),
                "width": stream["width"], "height": stream["height"],
                "fps": stream["avg_frame_rate"], "frames": len(timestamps),
                "codec": stream["codec_name"]}
    if packets:
        packet_list = data.get("packets", [item for item in entries
                                          if item.get("type") == "packet"])
        metadata["video_bps"] = float(8 * sum(int(p["size"]) for p in packet_list) / duration)
        metadata["file_bps"] = float(8 * metadata["bytes"] / duration)
    return metadata, duration, timestamps


def check_output(args, path, source, source_duration, source_times):
    run([args.ffmpeg, "-v", "error", "-xerror", "-i", str(path),
         "-map", "0:v:0", "-f", "null", os.devnull])
    info, duration, times = probe(args.ffprobe, path, packets=True)
    for field in ("width", "height", "fps", "frames"):
        if info[field] != source[field]:
            raise ValueError(f"Encoded {field} differs from source")
    tolerance = Fraction(1, 1000)
    if abs(duration - source_duration) > tolerance or any(
        abs((out - times[0]) - (src - source_times[0])) > tolerance
        for out, src in zip(times, source_times)
    ):
        raise ValueError("Encoded timeline differs from source")
    return info


def choose_settings(selector, source, metadata, target_bytes):
    if selector is None:
        return BASELINE.copy(), {"reason": "Fixed baseline settings"}
    decision = selector(source, metadata.copy(), target_bytes)
    if not isinstance(decision, dict):
        raise ValueError("Selector must return a dictionary")
    preset = decision.get("preset")
    mode = decision.get("aq_mode")
    if not isinstance(preset, str) or preset not in PRESETS or type(mode) is not int or mode not in range(4):
        raise ValueError("Selector returned an unsupported preset or AQ mode")
    return {"preset": preset, "aq_mode": mode}, decision


def encode_job(args, source, percent, method, selector, evaluator, output_dir):
    started = perf_counter()
    record = {"video": source.name, "method": method, "target_percent": percent,
              "attempts": [], "status": "failed", "output": None}
    output_path = output_dir / f"{source.stem}.mp4"
    try:
        metadata, duration, timestamps = probe(args.ffprobe, source)
        target_bytes = metadata["bytes"] * percent // 100
        target_bps = 8 * target_bytes / duration
        record.update(source=metadata, target_bytes=target_bytes, target_bps=float(target_bps))
        settings, decision = choose_settings(selector, source, metadata, target_bytes)
        json.dumps(decision, allow_nan=False)
        record.update(settings=settings, decision=decision)
        bitrate = max(1000, int(target_bps * Fraction(98, 100)))
        with TemporaryDirectory(prefix="encode-", dir=output_dir) as work:
            for number in range(1, args.attempts + 1):
                candidate = Path(work) / f"attempt-{number}.mp4"
                passlog = Path(work) / f"pass-{number}"
                common = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                          "-y", "-i", str(source), "-map", "0:v:0", "-an",
                          "-fps_mode", "passthrough", "-c:v", "libx264",
                          "-preset", settings["preset"], "-pix_fmt", "yuv420p",
                          "-threads", str(args.threads), "-b:v", str(bitrate),
                          "-x264-params", f"aq-mode={settings['aq_mode']}",
                          "-passlogfile", str(passlog)]
                first = [*common, "-pass", "1", "-f", "null", os.devnull]
                second = [*common, "-pass", "2", "-movflags", "+faststart", str(candidate)]
                attempt = {"number": number, "requested_bps": bitrate,
                           "commands": [first, second]}
                record["attempts"].append(attempt)
                attempt_started = perf_counter()
                try:
                    run(first)
                    run(second)
                    attempt["encode_s"] = perf_counter() - attempt_started
                    actual_bytes = candidate.stat().st_size
                    attempt["actual_bytes"] = actual_bytes
                    if actual_bytes > target_bytes:
                        next_bitrate = max(1000, bitrate * target_bytes * 98 // (actual_bytes * 100))
                        attempt["reason"] = "Output exceeds the byte budget"
                        if number == args.attempts or next_bitrate >= bitrate:
                            raise ValueError("Unable to meet byte budget within the attempt limit")
                        attempt["next_requested_bps"] = next_bitrate
                        bitrate = next_bitrate
                        continue
                    result = check_output(args, candidate, metadata, duration, timestamps)
                    attempt["decode_ok"] = True
                    attempt["result"] = result
                    if evaluator is not None:
                        metrics = evaluator(source, candidate)
                        json.dumps(metrics, allow_nan=False)
                        record["quality"] = metrics
                    candidate.replace(output_path)
                    record.update(status="ok", output=str(output_path),
                                  selected_attempt=number, result=result)
                    break
                except (OSError, RuntimeError, ValueError) as error:
                    attempt["error"] = str(error)
                    raise
                finally:
                    attempt["elapsed_s"] = perf_counter() - attempt_started
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["elapsed_s"] = perf_counter() - started
    return record


def load_hook(spec):
    module, separator, function = spec.partition(":")
    if not separator or not module or not function:
        raise ValueError("Hooks must use module:function syntax")
    hook = getattr(importlib.import_module(module), function)
    if not callable(hook):
        raise ValueError(f"{spec} is not callable")
    return hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One MP4 or a dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--mode", choices=["baseline", "adaptive", "both"], default="baseline")
    parser.add_argument("--selector", help="Adaptive parameter hook, module:function")
    parser.add_argument("--evaluator", help="Quality evaluation hook, module:function")
    parser.add_argument("--attempts", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ffmpeg", default=os.environ.get("CS5185_FFMPEG", "ffmpeg"))
    parser.add_argument("--ffprobe", default=os.environ.get("CS5185_FFPROBE", "ffprobe"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.mode != "baseline" and not args.selector:
        parser.error("adaptive mode requires --selector")
    path = args.input.resolve()
    videos = sorted(path.glob("*.mp4")) if path.is_dir() else [path]
    if not videos or any(not video.is_file() or video.suffix.lower() != ".mp4" for video in videos):
        parser.error("Input must be an MP4 or a directory containing MP4 files")
    args.output = args.output.resolve()
    try:
        selector = load_hook(args.selector) if args.mode != "baseline" else None
        evaluator = load_hook(args.evaluator) if args.evaluator else None
        if args.evaluator == "quality:evaluate":
            evaluator = partial(evaluator, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
        ffmpeg_version = run([args.ffmpeg, "-version"]).splitlines()[0]
        ffprobe_version = run([args.ffprobe, "-version"]).splitlines()[0]
        encoders = run([args.ffmpeg, "-hide_banner", "-encoders"])
        if not any(line.split()[1:2] == ["libx264"] for line in encoders.splitlines()):
            raise ValueError("FFmpeg must include the libx264 encoder")
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, RuntimeError, ValueError, ImportError, AttributeError) as error:
        parser.error(str(error))
    methods = ["baseline", "adaptive"] if args.mode == "both" else [args.mode]
    config = {"ffmpeg": ffmpeg_version, "ffprobe": ffprobe_version,
              "threads": args.threads, "attempt_limit": args.attempts, "methods": methods,
              "selector": args.selector, "evaluator": args.evaluator, "input": str(path)}
    (args.output / "run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    failures = 0
    with (args.output / "jobs.jsonl").open("w", encoding="utf-8") as log:
        for method in methods:
            for percent in (5, 2):
                directory = args.output / method / str(percent)
                directory.mkdir(parents=True)
                for source in videos:
                    hook = selector if method == "adaptive" else None
                    record = encode_job(args, source, percent, method, hook, evaluator, directory)
                    log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    failures += record["status"] != "ok"
                    print(f"{method} {percent}% {source.name}: {record['status']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

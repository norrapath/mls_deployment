import argparse
import os
import subprocess
import sys

import numpy as np


SAMPLE_RATE = 16000
FRAME_SECONDS = 0.05
STD_MULTIPLIER = 1.0
PADDING_SECONDS = 0.15
MIN_VOICE_SECONDS = 0.10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-detect voice/sound by frame dB statistics, trim the file, and save it."
    )
    parser.add_argument("input_file", help="Audio or video file to trim.")
    parser.add_argument(
        "output_target", 
        help="Target output folder OR complete output file path."
    )
    parser.add_argument(
        "--std-multiplier",
        type=float,
        default=STD_MULTIPLIER,
        help="Voice threshold = median frame dB + this value * dB standard deviation.",
    )
    parser.add_argument(
        "--frame-seconds",
        type=float,
        default=FRAME_SECONDS,
        help="Analysis frame size in seconds.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=PADDING_SECONDS,
        help="Seconds to keep before first detected voice and after last detected voice.",
    )
    parser.add_argument(
        "--min-voice-seconds",
        type=float,
        default=MIN_VOICE_SECONDS,
        help="Minimum detected voice duration required before trimming.",
    )
    return parser.parse_args()


def get_duration(filename):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    return float(result.stdout.strip())


def load_audio_samples(filename):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        filename,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def frame_db(samples, frame_seconds):
    frame_size = max(1, int(SAMPLE_RATE * frame_seconds))
    frame_count = len(samples) // frame_size

    if frame_count == 0:
        return np.array([], dtype=np.float32)

    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-12))


def detect_voice_bounds(filename, frame_seconds, std_multiplier, padding, min_voice_seconds):
    duration = get_duration(filename)
    samples = load_audio_samples(filename)
    db_values = frame_db(samples, frame_seconds)

    if db_values.size == 0:
        print("[WARNING]: No audio samples loaded. Fallback to full duration.", file=sys.stderr)
        return 0.0, duration, 0.0, 0.0, 0.0

    db_median = float(np.median(db_values))
    db_std = float(np.std(db_values))
    threshold = db_median + (std_multiplier * db_std)
    voice_frames = np.flatnonzero(db_values > threshold)

    if voice_frames.size == 0:
        print(
            f"[WARNING]: No distinct voice detected (Median={db_median:.2f} dB, threshold={threshold:.2f} dB). "
            "Fallback to full file duration.",
            file=sys.stderr,
        )
        return 0.0, duration, threshold, db_median, db_std

    start = max(0.0, float(voice_frames[0]) * frame_seconds - padding)
    end = min(duration, (float(voice_frames[-1]) + 1.0) * frame_seconds + padding)

    if end - start < min_voice_seconds:
        print(
            f"[WARNING]: Detected voice duration ({end - start:.2f}s) is below minimum threshold. "
            "Fallback to full file duration.",
            file=sys.stderr,
        )
        return 0.0, duration, threshold, db_median, db_std

    return start, end, threshold, db_median, db_std


def resolve_output_path(input_file, output_target):
    # Check if target is explicitly a directory or lacks an extension
    if os.path.isdir(output_target) or not os.path.splitext(output_target)[1]:
        os.makedirs(output_target, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(input_file))
        return os.path.join(output_target, f"{name}_trimmed{ext}")
    else:
        parent_dir = os.path.dirname(output_target)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return output_target


def trim_file(input_file, output_file, start, end):
    duration = end - start
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        input_file,
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        output_file,
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    try:
        start, end, threshold, db_median, db_std = detect_voice_bounds(
            args.input_file,
            args.frame_seconds,
            args.std_multiplier,
            args.padding,
            args.min_voice_seconds,
        )
        output_file = resolve_output_path(args.input_file, args.output_target)
        trim_file(args.input_file, output_file, start, end)
    except Exception as exc:
        print(f"Error executing trim process: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Median dB: {db_median:.2f}")
    print(f"Std dB: {db_std:.2f}")
    print(f"Voice threshold: {threshold:.2f} dB")
    print(f"Trim range: {start:.2f}s -> {end:.2f}s")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
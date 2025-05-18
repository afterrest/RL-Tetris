import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = [
    "epoch/score",
    "epoch/cleared_lines",
    "epoch/memory_size",
    "train/td_loss",
    "train/q_value",
    "train/target_q_value",
    "schedule/epsilon",
]

# db scalar look up
def collect_scalars(logdir: str, tags):
    data = {tag: [] for tag in tags}
    event_files = glob(os.path.join(logdir, "**", "events.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No event files found under {logdir}")

    for ev_file in event_files:
        ea = EventAccumulator(ev_file, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in tags:
            if tag in ea.Tags().get("scalars", []):
                scalars = ea.Scalars(tag)
                data[tag].extend((s.step, s.value) for s in scalars)

    for tag in tags:
        seen = {}
        for step, val in sorted(data[tag], key=lambda t: t[0]):
            seen[step] = val         
        data[tag] = list(seen.items())
    return data


# get MA
def moving_average(values, window=20):
    import numpy as np

    if window <= 1:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    head = [sum(values[:i + 1]) / (i + 1) for i in range(window - 1)]
    return np.concatenate([head, ma])


def plot_scalar(steps, values, tag, outdir, smooth=False):
    fig = plt.figure()
    plt.plot(steps, values, label=tag)
    if smooth and len(values) > 20:
        plt.plot(steps, moving_average(values), linestyle="--",
                 label=f"{tag} (MA)")
    plt.xlabel("epoch")
    plt.ylabel(tag.split("/", 1)[-1])
    plt.title(tag)
    plt.legend()
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{tag.replace('/', '_')}.png")
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True,
                        help="tensorboard 로그 디렉토리 (runs/…)")
    parser.add_argument("--outdir", default="./figs",
                        help="PNG 파일 저장 경로")
    parser.add_argument("--tags", nargs="+", default=TAGS,
                        help="그릴 스칼라 태그 목록")
    parser.add_argument("--smooth", action="store_true",
                        help="이동 평균(창: 20) 보조선 추가")
    args = parser.parse_args()

    data = collect_scalars(args.logdir, args.tags)
    for tag, series in data.items():
        if not series:
            print(f"[WARN] tag '{tag}' not found, skip.")
            continue
        steps, values = zip(*series)
        plot_scalar(steps, values, tag, args.outdir, args.smooth)


if __name__ == "__main__":
    main()
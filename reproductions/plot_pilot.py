"""Plot saved pilot signals in their declared physical time and voltage units."""

import argparse
from pathlib import Path

import numpy as np

from reproductions.evaluate import kaggle_timeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-pilot", required=True)
    parser.add_argument("--second-pilot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first, second = Path(args.first_pilot), Path(args.second_pilot)
    case = "D0_000022_02"
    gold = np.load(second / "pilot-inputs/gold" / (case + ".npz"))
    native = np.load(second / "npj-d0-500hz" / (case + ".npz"))["values"] / 1000
    kaggle = kaggle_timeline(np.load(second / "kaggle-d0/native/digitalised" / (case + ".series.npy")))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    colors = {"reference": "#222222", "npj": "#b65a2b", "kaggle": "#187e8e"}
    for ax, index, title, limits in [(axes[0], 1, "D0 development record, lead II", (0, 3)), (axes[1], 6, "Same D0 record, printed V1 segment", (5, 7.5))]:
        t = np.arange(5000) / 500
        ax.plot(t, np.where(gold["printed"][index], gold["values"][index], np.nan), color=colors["reference"], lw=1.8, label="Reference")
        ax.plot(t, native[index], color=colors["npj"], lw=1, alpha=0.9, label="npj, explicit 5,000-sample output")
        ax.plot(t, kaggle[index], color=colors["kaggle"], lw=1, alpha=0.85, label="Kaggle native six-model ensemble")
        ax.set(xlim=limits, title=title, ylabel="mV")
    case = "D2_000001_p00"
    gold = np.load(first / "pilot-inputs/gold" / (case + ".npz"))
    native = np.load(first / "npj-precordial" / (case + ".npz"))["values"] / 1000
    t = np.arange(5000) / 1000
    axes[2].plot(t, gold["values"][11], color=colors["reference"], lw=1.5, label="Reference, V6 printed on source page")
    axes[2].plot(t, native[11], color=colors["npj"], label="npj V6, all samples missing")
    axes[2].text(0.98, 0.92, "Returned V6 samples: 0 / 5,000", transform=axes[2].transAxes, ha="right", va="top", color=colors["npj"])
    axes[2].set(xlim=(0, 5), title="Ahus development record, native V6 omission", ylabel="mV")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Two development records, not an article or leaderboard reproduction", fontsize=13)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

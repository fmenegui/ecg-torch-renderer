"""Plot the first manifest case on unchanged physical axes, without alignment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    case = json.loads((args.data / "manifest.json").read_text())["cases"][0]
    with np.load(args.data / case["reference"], allow_pickle=False) as z:
        gold = z["values_mv"].copy()
    signals = {}
    for arm in ["npj_native", "grid_adaptive"]:
        with np.load(args.run / case["image_id"] / (arm + ".npz"), allow_pickle=False) as z:
            signals[arm] = z["values_mv"].copy()
            if float(z["fs"]) != 500 or signals[arm].shape != gold.shape:
                raise ValueError("Inconsistent physical axes")
    t = np.arange(gold.shape[1]) / 500
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), sharex=True, constrained_layout=True)
    for column, lead in enumerate([0, 1]):
        finite = np.concatenate([v[lead][np.isfinite(v[lead])] for v in [gold, *signals.values()]])
        low, high = float(finite.min()), float(finite.max())
        margin = max(.05, (high-low)*.1)
        for row, (arm, label, color) in enumerate([
            ("npj_native", "npj direto", "#2563eb"),
            ("grid_adaptive", "Grade + RGB adaptativo + npj", "#15803d"),
        ]):
            ax = axes[row, column]
            ax.fill_between(t, low-margin, high+margin, where=~np.isfinite(gold[lead]), color="#f1f5f9", step="mid")
            ax.plot(t, gold[lead], color="#111827", lw=1.3, label="Referência")
            ax.plot(t, signals[arm][lead], color=color, lw=.9, alpha=.85, label=label)
            ax.set(title=("I" if lead == 0 else "II") + " · " + label, ylabel="mV", ylim=(low-margin, high+margin))
            ax.set_xlim(0, t[-1])
            ax.grid(alpha=.15)
            ax.legend(loc="upper right", fontsize=8)
            if row == 1:
                ax.set_xlabel("Tempo físico (s)")
    fig.suptitle("Primeiro caso do manifesto congelado: " + case["image_id"], fontsize=13)
    fig.supxlabel("Eixos brutos. Sem deslocamento, ajuste de ganho ou DC. Cinza: fora do intervalo impresso. NaNs mantidos. Uma imagem não resume o ensaio.", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

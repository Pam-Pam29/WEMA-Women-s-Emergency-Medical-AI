"""
WEMA — Women's Emergency Medical AI
reports/kb/generate_kb_charts.py

Regenerates kb_by_source.png and kb_chunks_per_source.png from
reports/kb/knowledge_base_manifest.csv, so the figures stay reproducible
from the manifest instead of being hand-edited images.

Run from the project root:
    python reports/kb/generate_kb_charts.py
"""

import os
import csv
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "knowledge_base_manifest.csv")

COLOR_SOURCE = "#d85a30"   # orange — documents by source
COLOR_CHUNKS = "#1d9e75"   # teal — chunks ingested per document

plt.rcParams["font.family"] = "DejaVu Sans"


def load_manifest():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


def plot_by_source(rows, out_path):
    counts = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    # sort descending by count, ties broken by first-appearance order in the manifest
    ordered_sources = []
    for row in rows:
        if row["source"] not in ordered_sources:
            ordered_sources.append(row["source"])
    ordered_sources.sort(key=lambda s: -counts[s])

    values = [counts[s] for s in ordered_sources]

    fig, ax = plt.subplots(figsize=(9.15, 6.11), dpi=150)
    bars = ax.bar(ordered_sources, values, color=COLOR_SOURCE, width=0.65)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
            str(v), ha="center", va="bottom", fontsize=13, color="#222222",
        )

    ax.set_title("WEMA Knowledge Base — Documents by Source", fontsize=20, fontweight="bold", pad=20)
    ax.set_ylabel("documents", fontsize=12, color="#595959")
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12, colors="#595959")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_chunks_per_source(rows, out_path):
    ordered = sorted(rows, key=lambda r: -int(r["chunks"]))
    titles = [r["title"] for r in ordered]
    chunks = [int(r["chunks"]) for r in ordered]

    fig, ax = plt.subplots(figsize=(14.4, 7.1), dpi=200)
    y_pos = range(len(titles))
    bars = ax.barh(y_pos, chunks, color=COLOR_CHUNKS, height=0.65)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(titles, fontsize=10)
    ax.invert_yaxis()  # largest at top, matching original layout

    max_val = max(chunks)
    for bar, v in zip(bars, chunks):
        ax.text(
            bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            str(v), ha="left", va="center", fontsize=10, color="#222222",
        )

    ax.set_title("WEMA Knowledge Base — Chunks Ingested per Document", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("chunks in ChromaDB", fontsize=11, color="#595959")
    ax.set_xlim(0, max_val * 1.12)
    ax.tick_params(axis="x", labelsize=10, colors="#595959")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    rows = load_manifest()
    total = sum(int(r["chunks"]) for r in rows)
    print(f"Loaded {len(rows)} documents, {total:,} total chunks from manifest.")

    by_source_path = os.path.join(HERE, "kb_by_source.png")
    chunks_path = os.path.join(HERE, "kb_chunks_per_source.png")

    plot_by_source(rows, by_source_path)
    print(f"Wrote {by_source_path}")

    plot_chunks_per_source(rows, chunks_path)
    print(f"Wrote {chunks_path}")


if __name__ == "__main__":
    main()

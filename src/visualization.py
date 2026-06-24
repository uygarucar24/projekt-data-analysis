"""Visualisierung der Ergebnisse.

Erzeugt und speichert (in ``results/``):
- Balkendiagramme der wichtigsten Terme (BoW vs. TF-IDF),
- Wortwolken je extrahiertem Thema,
- Balkendiagramm der Themenverteilung über die Dokumente.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nicht-interaktives Backend (Speichern ohne Anzeige)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from . import config


def _ensure_results_dir() -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.RESULTS_DIR


def plot_top_terms(top_bow, top_tfidf, filename: str = "01_vergleich_vektorisierung.png") -> Path:
    """Vergleichende Balkendiagramme der wichtigsten Terme beider Verfahren."""
    out = _ensure_results_dir() / filename
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in (
        (axes[0], top_bow, "Bag-of-Words: häufigste Terme"),
        (axes[1], top_tfidf, "TF-IDF: höchstgewichtete Terme"),
    ):
        terms = [t for t, _ in data][::-1]
        scores = [s for _, s in data][::-1]
        ax.barh(terms, scores, color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel("Gesamtgewicht")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def wordclouds_for_topics(topics: list[list[str]], method: str) -> Path:
    """Erzeugt für jedes Thema eine Wortwolke (in einer Abbildung zusammengefasst)."""
    out = _ensure_results_dir() / f"02_wortwolken_{method.lower()}.png"
    n = len(topics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    axes = axes.ravel() if n > 1 else [axes]

    for i, words in enumerate(topics):
        # Gewichtung über die Rangfolge (vorderste Wörter größer)
        freqs = {w: len(words) - j for j, w in enumerate(words)}
        wc = WordCloud(width=500, height=300, background_color="white",
                       colormap="viridis").generate_from_frequencies(freqs)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"{method} – Thema {i + 1}", fontsize=11)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_topic_distribution(counts_lda, counts_nmf, filename: str = "03_themenverteilung.png") -> Path:
    """Balkendiagramm: Anzahl Dokumente je dominantem Thema (LDA vs. NMF)."""
    out = _ensure_results_dir() / filename
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, counts, title, color in (
        (axes[0], counts_lda, "LDA – Dokumente je Thema", "#55A868"),
        (axes[1], counts_nmf, "NMF – Dokumente je Thema", "#C44E52"),
    ):
        themen = [f"T{t + 1}" for t in sorted(counts)]
        werte = [counts[t] for t in sorted(counts)]
        ax.bar(themen, werte, color=color)
        ax.set_title(title)
        ax.set_ylabel("Anzahl Dokumente")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out

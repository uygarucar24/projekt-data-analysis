"""Themen-Extraktion (semantische Analyse).

Es werden ZWEI semantische Analysetechniken umgesetzt und verglichen
(Aufgabenanforderung), ergänzt um eine dritte als Bonus:

1. LDA  – Latent Dirichlet Allocation   (probabilistisch, auf Bag-of-Words)
2. NMF  – Non-negative Matrix Factorization (linear-algebraisch, auf TF-IDF)
3. LSA  – Latent Semantic Analysis (TruncatedSVD, auf TF-IDF) — optionaler Vergleich
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

from . import config


def fit_lda(bow_matrix, n_topics: int = config.DEFAULT_N_TOPICS):
    """Latent Dirichlet Allocation auf Bag-of-Words-Zählmatrix."""
    model = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=20,
        random_state=config.RANDOM_STATE,
    )
    doc_topic = model.fit_transform(bow_matrix)
    return model, doc_topic


def fit_nmf(tfidf_matrix, n_topics: int = config.DEFAULT_N_TOPICS):
    """Non-negative Matrix Factorization auf TF-IDF-Matrix."""
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        max_iter=400,
        random_state=config.RANDOM_STATE,
    )
    doc_topic = model.fit_transform(tfidf_matrix)
    return model, doc_topic


def fit_lsa(tfidf_matrix, n_topics: int = config.DEFAULT_N_TOPICS):
    """Latent Semantic Analysis (TruncatedSVD) auf TF-IDF-Matrix."""
    model = TruncatedSVD(n_components=n_topics, random_state=config.RANDOM_STATE)
    doc_topic = model.fit_transform(tfidf_matrix)
    return model, doc_topic


def get_top_words(model, feature_names, n_top: int = config.DEFAULT_N_TOP_WORDS) -> list[list[str]]:
    """Liefert je Thema die ``n_top`` Wörter mit dem höchsten Gewicht."""
    topics = []
    for comp in model.components_:
        top_idx = comp.argsort()[::-1][:n_top]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def dominant_topic_counts(doc_topic) -> dict[int, int]:
    """Zählt, wie vielen Dokumenten jedes Thema als dominantes Thema zugeordnet ist."""
    dominant = np.asarray(doc_topic).argmax(axis=1)
    unique, counts = np.unique(dominant, return_counts=True)
    return {int(t): int(c) for t, c in zip(unique, counts)}


def format_topics(topics: list[list[str]], title: str) -> str:
    """Erzeugt eine lesbare Textdarstellung der Themen."""
    lines = [f"### {title}"]
    for i, words in enumerate(topics):
        lines.append(f"Thema {i + 1}: " + ", ".join(words))
    return "\n".join(lines)

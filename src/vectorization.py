"""Vektorisierung: Umwandlung der sauberen Texte in numerische Vektoren.

Es werden ZWEI Techniken umgesetzt und verglichen (Aufgabenanforderung):
1. Bag-of-Words   (CountVectorizer)  -> reine Häufigkeiten
2. TF-IDF         (TfidfVectorizer)  -> häufigkeitsgewichtete, selten-belohnende Gewichte
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config


def build_bow(docs: list[str], min_df: int = config.MIN_DF,
              max_df: float = config.MAX_DF, max_features: int = config.MAX_FEATURES):
    """Bag-of-Words-Vektorisierung (absolute Termhäufigkeiten)."""
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
    matrix = vectorizer.fit_transform(docs)
    return matrix, vectorizer


def build_tfidf(docs: list[str], min_df: int = config.MIN_DF,
                max_df: float = config.MAX_DF, max_features: int = config.MAX_FEATURES):
    """TF-IDF-Vektorisierung (term frequency * inverse document frequency)."""
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
    matrix = vectorizer.fit_transform(docs)
    return matrix, vectorizer


def top_terms(vectorizer, matrix, n: int = 15) -> list[tuple[str, float]]:
    """Liefert die ``n`` Terme mit dem höchsten Gesamtgewicht über alle Dokumente."""
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = scores.argsort()[::-1][:n]
    return [(terms[i], float(scores[i])) for i in order]


def compare_vectorizers(bow_vec, bow_mat, tfidf_vec, tfidf_mat) -> dict:
    """Erstellt eine kurze Vergleichsübersicht der beiden Vektorisierungen."""
    def sparsity(m) -> float:
        return 1.0 - (m.nnz / (m.shape[0] * m.shape[1]))

    return {
        "bow": {
            "vokabular": len(bow_vec.get_feature_names_out()),
            "matrix_shape": bow_mat.shape,
            "sparsity": round(sparsity(bow_mat), 4),
            "top_terme": top_terms(bow_vec, bow_mat, 10),
        },
        "tfidf": {
            "vokabular": len(tfidf_vec.get_feature_names_out()),
            "matrix_shape": tfidf_mat.shape,
            "sparsity": round(sparsity(tfidf_mat), 4),
            "top_terme": top_terms(tfidf_vec, tfidf_mat, 10),
        },
    }

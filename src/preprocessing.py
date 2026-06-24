"""Textvorverarbeitung ("saubere Texte" erzeugen).

Schritte:
1. Bereinigung per Regex (URLs, @-Mentions, #-Symbole, Zahlen, Sonderzeichen).
2. Tokenisierung + Lemmatisierung mit spaCy (deutsches Modell ``de_core_news_sm``).
3. Entfernen von Stoppwörtern und sehr kurzen Tokens.

Ist das spaCy-Modell nicht installiert, wird automatisch auf eine leichtgewichtige
Regex-Tokenisierung mit Stoppwortliste zurückgegriffen, damit der Code lauffähig bleibt.
"""
from __future__ import annotations

import re
from typing import Iterable

from . import config

# --------------------------------------------------------------------------- #
# spaCy laden (optional). Parser & NER werden zur Beschleunigung deaktiviert,
# da wir nur Lemmatisierung und Wortarten benötigen.
# --------------------------------------------------------------------------- #
try:
    import spacy
    from spacy.lang.de.stop_words import STOP_WORDS as SPACY_STOPWORDS

    _NLP = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    _SPACY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _NLP = None
    _SPACY_AVAILABLE = False
    # Minimale deutsche Stoppwortliste als Fallback (Auszug der häufigsten Begriffe)
    SPACY_STOPWORDS = {
        "der", "die", "das", "und", "in", "zu", "den", "mit", "von", "auf", "für",
        "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch", "es", "an",
        "werden", "aus", "er", "hat", "sind", "oder", "am", "sein", "einem",
        "einer", "wie", "wir", "was", "ich", "du", "sie", "man", "aber", "nach",
        "wird", "bei", "noch", "über", "so", "zum", "war", "haben", "nur", "um",
        "vom", "zur", "bis", "mehr", "durch", "kann", "sich", "dass", "diese",
        "dieser", "schon", "wenn", "weil", "ja", "nein", "doch", "vor", "seit",
    }

STOPWORDS = set(SPACY_STOPWORDS) | set(config.EXTRA_STOPWORDS)

# Regex-Muster für die Grobreinigung
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NONALPHA_RE = re.compile(r"[^a-zA-Zäöüß\s]")  # nur Buchstaben + Umlaute behalten
_MULTISPACE_RE = re.compile(r"\s+")


class GermanTextPreprocessor:
    """Vorverarbeitung deutschsprachiger Texte zu Listen sauberer Lemmata."""

    def __init__(self, min_token_len: int = 3, use_spacy: bool = True):
        self.min_token_len = min_token_len
        self.use_spacy = use_spacy and _SPACY_AVAILABLE
        self.backend = "spaCy (de_core_news_sm)" if self.use_spacy else "Regex-Fallback"

    # ------------------------------------------------------------------ #
    def clean(self, text: str) -> str:
        """Grobreinigung: Kleinschreibung, Entfernen von URLs/Mentions/Sonderzeichen."""
        text = text.lower()
        text = _URL_RE.sub(" ", text)
        text = _MENTION_RE.sub(" ", text)
        text = text.replace("#", " ")
        text = _NONALPHA_RE.sub(" ", text)
        text = _MULTISPACE_RE.sub(" ", text).strip()
        return text

    # ------------------------------------------------------------------ #
    def _keep_token(self, token: str) -> bool:
        return len(token) >= self.min_token_len and token not in STOPWORDS

    def _tokens_spacy(self, doc) -> list[str]:
        tokens = []
        for tok in doc:
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            lemma = tok.lemma_.lower().strip()
            if self._keep_token(lemma):
                tokens.append(lemma)
        return tokens

    def _tokens_regex(self, cleaned: str) -> list[str]:
        return [t for t in cleaned.split() if self._keep_token(t)]

    # ------------------------------------------------------------------ #
    def preprocess_corpus(self, texts: Iterable[str]) -> list[str]:
        """Verarbeitet eine Sammlung von Texten.

        Rückgabe: Liste von Strings, in denen die sauberen Lemmata durch
        Leerzeichen getrennt sind (direkt für scikit-learn-Vektorisierer nutzbar).
        """
        cleaned = [self.clean(t) for t in texts]

        if self.use_spacy:
            docs = []
            # nlp.pipe verarbeitet effizient im Batch
            for doc in _NLP.pipe(cleaned, batch_size=200):
                docs.append(" ".join(self._tokens_spacy(doc)))
            return docs

        return [" ".join(self._tokens_regex(c)) for c in cleaned]

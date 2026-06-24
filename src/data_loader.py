"""Datenbeschaffung.

Lädt die reale Beschwerde-/Feedback-Textsammlung (GermEval 2017) aus dem Internet
und fällt bei fehlender Verbindung automatisch auf den mitgelieferten
Offline-Beispieldatensatz zurück. So ist die Analyse jederzeit reproduzierbar.
"""
from __future__ import annotations

import sys

import pandas as pd

from . import config


def _load_germeval(max_docs: int | None) -> pd.DataFrame:
    """Lädt GermEval 2017 (train + test) von der Hugging-Face-CDN."""
    frames = []
    for url in config.GERMEVAL_URLS:
        frames.append(pd.read_csv(url, encoding="utf-8"))
    df = pd.concat(frames, ignore_index=True)

    # Nur Spalten behalten, die wir benötigen; Text vereinheitlichen.
    df = df.rename(columns={config.TEXT_COLUMN: "text"})
    keep = [c for c in ("text", "relevance", "sentiment") if c in df.columns]
    df = df[keep].copy()
    df["quelle"] = "germeval2017"
    return df


def _load_sample(max_docs: int | None) -> pd.DataFrame:
    """Lädt den mitgelieferten deutschsprachigen Beispieldatensatz."""
    df = pd.read_csv(config.SAMPLE_CSV, encoding="utf-8")
    df["quelle"] = "beispiel"
    return df


def load_data(
    source: str = config.DEFAULT_SOURCE,
    max_docs: int | None = config.DEFAULT_MAX_DOCS,
) -> pd.DataFrame:
    """Liefert eine bereinigte Tabelle mit mindestens der Spalte ``text``.

    Parameters
    ----------
    source : {"germeval", "sample"}
        Datenquelle. Bei "germeval" wird online geladen; schlägt das fehl,
        erfolgt automatisch ein Fallback auf den Beispieldatensatz.
    max_docs : int | None
        Optionale Begrenzung der Dokumentanzahl (zufällige Stichprobe) zur
        Steuerung der Laufzeit. ``None`` = alle Dokumente verwenden.
    """
    if source == "germeval":
        try:
            df = _load_germeval(max_docs)
            print(f"[Daten] GermEval 2017 online geladen: {len(df)} Dokumente.")
        except Exception as exc:  # noqa: BLE001 - bewusst breit für robusten Fallback
            print(
                f"[Daten] Online-Download fehlgeschlagen ({type(exc).__name__}). "
                f"Nutze Offline-Beispieldatensatz.",
                file=sys.stderr,
            )
            df = _load_sample(max_docs)
    elif source == "sample":
        df = _load_sample(max_docs)
        print(f"[Daten] Beispieldatensatz geladen: {len(df)} Dokumente.")
    else:
        raise ValueError(f"Unbekannte Quelle: {source!r} (erlaubt: 'germeval', 'sample').")

    # Grundlegende Bereinigung der Tabelle (nicht des Textes!)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    if max_docs is not None and len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print(f"[Daten] Auf {max_docs} Dokumente reduziert (Stichprobe).")

    return df

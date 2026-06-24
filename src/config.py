"""Zentrale Konfiguration des Analyse-Projekts.

Alle Pfade und Standard-Parameter werden hier gebündelt, damit sie an einer
einzigen Stelle angepasst werden können (gute Praxis für reproduzierbare Analysen).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Verzeichnisse
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Mitgelieferter Offline-Beispieldatensatz (deutschsprachige Beschwerden)
SAMPLE_CSV = DATA_DIR / "beispiel_beschwerden_de.csv"

# --------------------------------------------------------------------------- #
# Datenquelle: GermEval 2017 (reales deutsches Kund:innen-Feedback zur Deutschen
# Bahn), gespiegelt auf Hugging Face und als CSV direkt abrufbar.
# Quelle: Wojatzki et al. (2017), "GermEval 2017: Shared Task on Aspect-based
# Sentiment in Social Media Customer Feedback".
# --------------------------------------------------------------------------- #
GERMEVAL_URLS = [
    "https://huggingface.co/datasets/akash418/germeval_2017/resolve/main/train.csv",
    "https://huggingface.co/datasets/akash418/germeval_2017/resolve/main/test.csv",
]
# Spaltenname mit dem eigentlichen Text in der Datenquelle
TEXT_COLUMN = "text"

# --------------------------------------------------------------------------- #
# Standard-Parameter der Analyse
# --------------------------------------------------------------------------- #
DEFAULT_SOURCE = "germeval"      # "germeval" oder "sample"
DEFAULT_MAX_DOCS = 5000          # Begrenzung der Dokumentanzahl (Laufzeit); None = alle
DEFAULT_N_TOPICS = 6             # Anzahl der zu extrahierenden Themen
DEFAULT_N_TOP_WORDS = 12         # Top-Wörter je Thema in der Ausgabe
RANDOM_STATE = 42                # Reproduzierbarkeit

# Vektorisierungs-Parameter
MIN_DF = 5                       # Term muss in >= 5 Dokumenten vorkommen
MAX_DF = 0.5                     # Term darf in <= 50 % der Dokumente vorkommen
MAX_FEATURES = 2000              # maximale Vokabulargröße

# Zusätzliche Stoppwörter: allgemein wenig informative Begriffe ...
EXTRA_STOPWORDS = {
    "dass", "schon", "mehr", "immer", "mal", "geht", "gibt", "wurde", "wurden",
    "heute", "gerade", "via", "rt", "amp", "http", "https", "uhr", "ab",
    "hab", "nen", "nem", "wer", "wo", "warum", "eigentlich",
    # ... sowie domänenspezifisches Rauschen des GermEval-Datensatzes
    # (allgegenwärtige Begriffe, die keine eigenständigen Themen darstellen):
    "deutsch", "deutsche", "deutschen", "bahn", "db", "deutschland",
    "gutefrage", "net", "dpa", "ticker",
}

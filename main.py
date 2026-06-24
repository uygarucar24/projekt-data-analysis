"""Hauptskript: vollständiger NLP-Analyse-Workflow für Bürger:innen-Beschwerden.

Ablauf (entspricht den Schritten der Aufgabenstellung):
    1. Daten laden            (data_loader)
    2. Texte vorverarbeiten   (preprocessing)
    3. Vektorisieren x2       (vectorization: Bag-of-Words & TF-IDF)
    4. Themen extrahieren x2  (topic_modeling: LDA & NMF, optional LSA)
    5. Ergebnisse visualisieren + zusammenfassen (visualization)

Aufruf (Beispiele):
    python main.py
    python main.py --source germeval --max-docs 5000 --n-topics 6
    python main.py --source sample
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime

# UTF-8-Ausgabe erzwingen (deutsche Umlaute auf der Windows-Konsole)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from src import config, data_loader, preprocessing, topic_modeling as tm, vectorization as vec
from src import visualization as viz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NLP-Themenanalyse deutschsprachiger Beschwerden.")
    p.add_argument("--source", choices=["germeval", "sample"], default=config.DEFAULT_SOURCE,
                   help="Datenquelle (Standard: %(default)s).")
    p.add_argument("--max-docs", type=int, default=config.DEFAULT_MAX_DOCS,
                   help="Maximale Dokumentanzahl; 0 = alle (Standard: %(default)s).")
    p.add_argument("--n-topics", type=int, default=config.DEFAULT_N_TOPICS,
                   help="Anzahl der Themen (Standard: %(default)s).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_docs = None if args.max_docs in (0, None) else args.max_docs

    report: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report.append(msg)

    log("=" * 70)
    log("NLP-THEMENANALYSE VON BÜRGER:INNEN-BESCHWERDEN")
    log(f"Lauf vom {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 70)

    # 1) Daten -------------------------------------------------------------
    df = data_loader.load_data(source=args.source, max_docs=max_docs)
    log(f"\n[1] Daten geladen: {len(df)} Dokumente aus Quelle '{df['quelle'].iloc[0]}'.")

    # 2) Vorverarbeitung ---------------------------------------------------
    pre = preprocessing.GermanTextPreprocessor()
    log(f"[2] Vorverarbeitung mit Backend: {pre.backend}")
    docs = pre.preprocess_corpus(df["text"].tolist())
    # leere Ergebnisse (nur Stoppwörter) verwerfen
    docs = [d for d in docs if d.strip()]
    log(f"    {len(docs)} nicht-leere Dokumente nach der Bereinigung.")

    # 3) Vektorisierung (zwei Techniken) -----------------------------------
    # Parameter an die Korpusgröße anpassen: bei kleinen Datensätzen (z. B. dem
    # Beispieldatensatz) muss min_df kleiner sein, sonst bleibt das Vokabular leer.
    if len(docs) < 500:
        min_df, max_df = 2, 0.95
    else:
        min_df, max_df = config.MIN_DF, config.MAX_DF

    bow_mat, bow_vec = vec.build_bow(docs, min_df=min_df, max_df=max_df)
    tfidf_mat, tfidf_vec = vec.build_tfidf(docs, min_df=min_df, max_df=max_df)
    cmp = vec.compare_vectorizers(bow_vec, bow_mat, tfidf_vec, tfidf_mat)

    log("\n[3] Vektorisierung – Vergleich:")
    log(f"    Bag-of-Words : Vokabular={cmp['bow']['vokabular']}, "
        f"Matrix={cmp['bow']['matrix_shape']}, Sparsity={cmp['bow']['sparsity']}")
    log(f"    TF-IDF       : Vokabular={cmp['tfidf']['vokabular']}, "
        f"Matrix={cmp['tfidf']['matrix_shape']}, Sparsity={cmp['tfidf']['sparsity']}")
    log("    Top-Terme BoW   : " + ", ".join(t for t, _ in cmp["bow"]["top_terme"]))
    log("    Top-Terme TF-IDF: " + ", ".join(t for t, _ in cmp["tfidf"]["top_terme"]))

    # 4) Themen-Extraktion (zwei Techniken + Bonus LSA) --------------------
    bow_terms = bow_vec.get_feature_names_out()
    tfidf_terms = tfidf_vec.get_feature_names_out()

    lda_model, lda_dt = tm.fit_lda(bow_mat, args.n_topics)
    nmf_model, nmf_dt = tm.fit_nmf(tfidf_mat, args.n_topics)
    lsa_model, _ = tm.fit_lsa(tfidf_mat, args.n_topics)

    lda_topics = tm.get_top_words(lda_model, bow_terms)
    nmf_topics = tm.get_top_words(nmf_model, tfidf_terms)
    lsa_topics = tm.get_top_words(lsa_model, tfidf_terms)

    log("\n[4] Extrahierte Themen:\n")
    log(tm.format_topics(lda_topics, "LDA (auf Bag-of-Words)"))
    log("")
    log(tm.format_topics(nmf_topics, "NMF (auf TF-IDF)"))
    log("")
    log(tm.format_topics(lsa_topics, "LSA / TruncatedSVD (auf TF-IDF) – Bonusvergleich"))

    counts_lda = tm.dominant_topic_counts(lda_dt)
    counts_nmf = tm.dominant_topic_counts(nmf_dt)

    # 5) Visualisierung ----------------------------------------------------
    f1 = viz.plot_top_terms(cmp["bow"]["top_terme"], cmp["tfidf"]["top_terme"])
    f2 = viz.wordclouds_for_topics(lda_topics, "LDA")
    f3 = viz.wordclouds_for_topics(nmf_topics, "NMF")
    f4 = viz.plot_topic_distribution(counts_lda, counts_nmf)

    log("\n[5] Abbildungen gespeichert:")
    for f in (f1, f2, f3, f4):
        log(f"    - {f.relative_to(config.PROJECT_ROOT)}")

    # Bericht schreiben ----------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.RESULTS_DIR / "ergebnisbericht.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    log(f"\n[OK] Textbericht gespeichert unter: {report_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

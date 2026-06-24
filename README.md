# NLP-Themenanalyse von Bürger:innen-Beschwerden

Portfolio-Projekt zum Kurs **„Projekt: Data Analysis" (DLBDSEDA02)** der IU Internationale Hochschule – **Aufgabe 1: NLP-Techniken anwenden, um eine Textsammlung zu analysieren.**

## 1. Ausgangssituation

Eine Kommune möchte besser auf die Sorgen und Beschwerden ihrer Bürger:innen eingehen. Über ein Online-System geht eine große Menge **unstrukturierter, freitextlicher Beschwerden** ein, die sich manuell nicht systematisch überblicken lässt. Ziel dieses Projekts ist es, mit Methoden der **maschinellen Sprachverarbeitung (NLP)** automatisch die **am häufigsten angesprochenen Themen** aus den Texten zu extrahieren und sie den Entscheidungstragenden übersichtlich bereitzustellen.

## 2. Datengrundlage

Als realer, frei verfügbarer Beispieldatensatz wird **GermEval 2017** verwendet: ~22.000 deutschsprachige Kund:innen-Rückmeldungen und Beschwerden über die Deutsche Bahn (öffentlicher Dienstleister). Der Datensatz bildet den Anwendungsfall – Beschwerden von Bürger:innen über einen öffentlichen Dienst – sehr gut ab.

> Wojatzki, M., Ruppert, E., Holschneider, S., Zesch, T., Biemann, C. (2017): *GermEval 2017 – Shared Task on Aspect-based Sentiment in Social Media Customer Feedback.* Verwendete Spiegelung (CSV): <https://huggingface.co/datasets/akash418/germeval_2017>.

Die Daten werden zur Laufzeit automatisch von der Hugging-Face-CDN geladen. **Ohne Internetverbindung** greift der Code automatisch auf einen mitgelieferten, kuratierten deutschsprachigen **Beispieldatensatz** (`data/beispiel_beschwerden_de.csv`, 75 typische Kommunal-Beschwerden) zurück, sodass die Analyse jederzeit reproduzierbar ist.

## 3. Methodisches Vorgehen

| Schritt | Verfahren | Bibliothek |
|--------|-----------|------------|
| 1. Datenbeschaffung | Laden von CSV (online/offline-Fallback) | `pandas` |
| 2. Vorverarbeitung | Bereinigung (Regex), Tokenisierung, **Lemmatisierung**, Stoppwortentfernung | `spaCy` (`de_core_news_sm`), `re` |
| 3. Vektorisierung (**2 Verfahren**) | **Bag-of-Words** und **TF-IDF** | `scikit-learn` |
| 4. Themen-Extraktion (**2 Verfahren** + Bonus) | **LDA**, **NMF**, ergänzend **LSA** | `scikit-learn` |
| 5. Visualisierung | Balkendiagramme, Wortwolken | `matplotlib`, `wordcloud` |

**Warum diese Verfahren?**
- *Bag-of-Words* zählt reine Worthäufigkeiten (einfach, interpretierbar). *TF-IDF* gewichtet zusätzlich die Seltenheit über Dokumente hinweg und unterdrückt allgegenwärtige, wenig informative Begriffe – ein aussagekräftiger Vergleich.
- *LDA* (Latent Dirichlet Allocation) ist ein **probabilistisches** Themenmodell auf Zähldaten; *NMF* (Non-negative Matrix Factorization) ist ein **linear-algebraisches** Verfahren, das auf TF-IDF in der Praxis oft schärfer trennbare Themen liefert. *LSA* (TruncatedSVD) ergänzt den Vergleich als klassisches latent-semantisches Verfahren.

## 4. Projektstruktur

```
projekt-data-analysis/
├── main.py                  # Orchestriert den gesamten Workflow (CLI)
├── requirements.txt         # pip-Abhängigkeiten
├── environment.yml          # alternative conda-Umgebung
├── data/
│   └── beispiel_beschwerden_de.csv   # Offline-Fallback-Datensatz (DE)
├── src/
│   ├── config.py            # zentrale Parameter & Pfade
│   ├── data_loader.py       # Schritt 1: Daten laden (+ Fallback)
│   ├── preprocessing.py     # Schritt 2: saubere Texte erzeugen
│   ├── vectorization.py     # Schritt 3: BoW & TF-IDF
│   ├── topic_modeling.py    # Schritt 4: LDA, NMF, LSA
│   └── visualization.py     # Schritt 5: Abbildungen
├── notebooks/
│   └── analyse.ipynb        # interaktive Schritt-für-Schritt-Analyse
├── docs/                    # Textentwürfe der Portfolio-Phasen
└── results/                 # erzeugte Abbildungen + Ergebnisbericht
```

## 5. Installation

**0. Repository klonen:**
```bash
git clone https://github.com/uygarucar24/projekt-data-analysis.git
cd projekt-data-analysis
```

**Variante A – venv + pip (empfohlen):**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

**Variante B – conda:**
```bash
conda env create -f environment.yml
conda activate beschwerden-nlp
```

## 6. Verwendung

```bash
# Standardlauf mit den realen GermEval-Daten (5.000 Dokumente, 6 Themen):
python main.py

# Mit dem mitgelieferten Offline-Beispieldatensatz:
python main.py --source sample --max-docs 0

# Parameter anpassen (alle Dokumente, 8 Themen):
python main.py --source germeval --max-docs 0 --n-topics 8
```

**Argumente:**
- `--source {germeval,sample}` – Datenquelle (Standard: `germeval`).
- `--max-docs N` – Stichprobengröße zur Steuerung der Laufzeit; `0` = alle.
- `--n-topics N` – Anzahl der zu extrahierenden Themen.

Ergebnisse landen im Ordner `results/`:
- `01_vergleich_vektorisierung.png` – Top-Terme Bag-of-Words vs. TF-IDF
- `02_wortwolken_lda.png` / `02_wortwolken_nmf.png` – Wortwolken je Thema
- `03_themenverteilung.png` – Dokumentanzahl je dominantem Thema
- `ergebnisbericht.md` – Textbericht des Laufs

## 7. Beispielhafte Ergebnisse (GermEval, 5.000 Dokumente)

Die mit **NMF** extrahierten Themen sind besonders trennscharf, u. a.:
- **Streik / Tarifkonflikt:** *streik, gdl, schlichtung, tarifkonflikt, gewerkschaft, lokführer, weselsky*
- **Verspätungen:** *zug, verspätung, minute, pünktlich, service, wagenreihung*
- **Strecken & Verkehrsmeldungen:** *neu, berlin, verkehrsmeldungen, hauptbahnhof, sperren, leipzig*
- **Fernbus / Alternativen:** *bus, fernbus, kostenlos, wlan, verbindung*

Der Vergleich zeigt: TF-IDF + NMF liefern hier schärfer abgegrenzte Themen als BoW + LDA, während LDA dafür weiche, überlappende Themenzugehörigkeiten je Dokument liefert. Für eine Kommune wären solche Themencluster eine direkte Priorisierungshilfe.

## 8. Lizenz / Hinweise

Der Code dient der Prüfungsleistung und steht zu Lern- und Demonstrationszwecken bereit. Die GermEval-2017-Daten unterliegen den Bedingungen der jeweiligen Herausgeber:innen und werden hier **nicht** mitgeliefert, sondern nur zur Laufzeit geladen.

# Portfolioteil 2 – Erläuterung der Implementierung (Erarbeitungs-/Reflexionsphase)

> **Hinweis:** Inhaltsentwurf (ca. 200 Wörter / ½ Seite). Vor der Abgabe als PDF
> formatieren und den **Link zum GitHub-Repository** eintragen.

**GitHub-Repository:** https://github.com/uygarucar24/projekt-data-analysis

## Umsetzung der Datenanalyse

Die Analyse wurde in Python modular umgesetzt (Paket `src/`, Orchestrierung über
`main.py`). Die Abhängigkeiten sind in einer virtuellen Umgebung installiert und
über `requirements.txt` bzw. `environment.yml` exportiert; der Code liegt
versionskontrolliert auf GitHub.

**Ablauf.** (1) Die realen Beschwerdetexte (GermEval 2017) werden per `pandas`
geladen; bei fehlender Verbindung greift automatisch ein mitgelieferter
Beispieldatensatz. (2) Die Vorverarbeitung erzeugt mit `spaCy` und Regex „saubere
Texte" (Lemmata ohne Stoppwörter). (3) Zur **Vektorisierung** werden
**Bag-of-Words** und **TF-IDF** (`scikit-learn`) gegenübergestellt: Beide ergeben
hier ein Vokabular von 2.000 Termen; TF-IDF dämpft allgegenwärtige Begriffe
sichtbar. (4) Zur **Themen-Extraktion** werden **LDA** (auf BoW) und **NMF** (auf
TF-IDF) – ergänzt um **LSA** – eingesetzt.

**Erste Ergebnisse.** NMF liefert die trennschärfsten Themen, u. a. *Streik/
Tarifkonflikt*, *Zug-Verspätungen*, *neue Strecken/Verkehrsmeldungen* und
*Fernbus-Alternativen*. LDA erzeugt weichere, leicht überlappende Themen.

**Aufgetretene Probleme.** Bei kleinen Korpora führte ein zu hohes `min_df` zu
leerem Vokabular – gelöst durch eine an die Korpusgröße angepasste Parametrisierung.

# Portfolioteil 1 – Konzept (Konzeptionsphase)

> **Hinweis:** Inhaltsentwurf (ca. 200 Wörter / ½ Seite). Bitte vor der Abgabe als
> PDF formatieren und ggf. an die eigene Formulierung anpassen.

## Konzept: Extraktion der häufigsten Themen aus Bürger:innen-Beschwerden

**Problem & Ziel.** Eine Kommune erhält über ein Online-Portal zahlreiche
unstrukturierte, freitextliche Beschwerden, die manuell nicht systematisch
auswertbar sind. Ziel ist es, mit NLP-Techniken automatisiert die am häufigsten
genannten Themen zu identifizieren und für Entscheidungstragende aufzubereiten.

**Datenquelle.** Verwendet wird der frei verfügbare, deutschsprachige Datensatz
*GermEval 2017* (~22.000 reale Kund:innen-Beschwerden über die Deutsche Bahn).
Er bildet den Anwendungsfall – Beschwerden über einen öffentlichen Dienst – gut
ab. Die Daten liegen als CSV mit einer Textspalte vor.

**Vorverarbeitung.** Kleinschreibung, Entfernen von URLs, Erwähnungen, Zahlen und
Sonderzeichen (Regex), anschließend Tokenisierung, **Lemmatisierung** und
Entfernung deutscher Stopp- sowie domänenspezifischer Rauschwörter.

**Vektorisierung (zwei Ansätze).** (1) **Bag-of-Words** (reine Häufigkeiten) und
(2) **TF-IDF** (häufigkeits- und seltenheitsgewichtet); beide werden verglichen.

**Themen-Extraktion (zwei Techniken).** (1) **LDA** (probabilistisch, auf BoW) und
(2) **NMF** (auf TF-IDF); ergänzend LSA als Bonusvergleich.

**Python-Bibliotheken.** `pandas` (Daten), `spaCy` + `re` (Vorverarbeitung),
`scikit-learn` (Vektorisierung, LDA/NMF/LSA), `matplotlib` + `wordcloud`
(Visualisierung).

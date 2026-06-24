# Portfolioteil 3 – Vollständige Zusammenfassung (Full Abstract)

> **Hinweis:** Inhaltsgerüst für die 2-seitige Zusammenfassung (Finalisierungsphase).
> Vor der Abgabe gemäß Formalia formatieren: **Arial 11 pt**, Überschriften 12 pt,
> **Blocksatz**, **1,5-Zeilenabstand**, Ränder 2 cm, Silbentrennung.
> Die **persönliche Reflexion** (Abschnitt 5) musst du selbst ausformulieren –
> nur du kennst deinen tatsächlichen Lernprozess.

---

## 1. Problemstellung und Zielsetzung

Eine Kommune möchte stärker auf die Anliegen ihrer Bürger:innen eingehen. Über ein
Online-Beschwerdeportal sammeln sich viele **unstrukturierte Freitexte**, die
manuell nicht überblickt werden können. Ziel des Projekts ist die **automatische
Extraktion der häufigsten Themen** aus diesen Beschwerden mithilfe von NLP, um
Entscheidungstragenden eine datenbasierte Priorisierung zu ermöglichen.

## 2. Daten und technischer Ansatz

Als realer Stellvertreterdatensatz dient **GermEval 2017** (~22.000 deutschsprachige
Beschwerden über die Deutsche Bahn). Der Workflow ist in Python modular aufgebaut:

1. **Datenbeschaffung** (`pandas`): Laden der CSV, automatischer Offline-Fallback.
2. **Vorverarbeitung** (`spaCy`, `re`): Bereinigung, Lemmatisierung, Stoppwortentfernung.
3. **Vektorisierung** (`scikit-learn`): **Bag-of-Words** und **TF-IDF** im Vergleich.
4. **Themen-Extraktion** (`scikit-learn`): **LDA**, **NMF** und ergänzend **LSA**.
5. **Visualisierung** (`matplotlib`, `wordcloud`): Top-Terme, Wortwolken, Themenverteilung.

## 3. Datenqualität

Die Texte sind kurz, umgangssprachlich und enthalten Tippfehler, URLs und
Social-Media-Artefakte. Die Vorverarbeitung und domänenspezifische Stoppwörter
(z. B. „bahn", „gutefrage") waren entscheidend, um aussagekräftige Themen zu
erhalten. Nicht alle Texte sind echte Beschwerden (auch neutrale Meldungen),
was die Themenbildung leicht verrauscht.

## 4. Ergebnisse und Diskussion

Beide Vektorisierungen liefern ein Vokabular von 2.000 Termen; **TF-IDF** dämpft
allgegenwärtige Begriffe und betont charakteristische Wörter. Bei der Themen-
Extraktion erweist sich **NMF auf TF-IDF** als am trennschärfsten. Identifizierte
Kernthemen: **Streik/Tarifkonflikt**, **Zug-Verspätungen**, **neue Strecken/
Verkehrsmeldungen** und **Fernbus-Alternativen**. **LDA** erzeugt weichere,
teils überlappende Themen mit probabilistischer Dokumentzuordnung. Für eine
Kommune ließen sich solche Cluster unmittelbar in Handlungsfelder übersetzen.

**Ausblick.** Verbesserungspotenzial besteht in: (a) Vorfilterung auf tatsächliche
Beschwerden via Sentiment-Klassifikation, (b) modernen Embeddings (z. B.
Sentence-BERT) statt BoW/TF-IDF, (c) automatischer Themenzahl-Bestimmung über
Kohärenzmaße sowie (d) einem interaktiven Dashboard für die Verwaltung.

## 5. Persönliche Reflexion *(bitte selbst ausformulieren)*

- Was habe ich auf persönlicher/fachlicher Ebene gelernt?
- Auf welche Probleme bin ich gestoßen und wie habe ich sie gelöst?
- Welche Problemlösungsstrategien nehme ich für künftige Projekte mit?

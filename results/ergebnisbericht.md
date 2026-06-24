======================================================================
NLP-THEMENANALYSE VON BÜRGER:INNEN-BESCHWERDEN
Lauf vom 2026-06-24 14:02
======================================================================

[1] Daten geladen: 5000 Dokumente aus Quelle 'germeval2017'.
[2] Vorverarbeitung mit Backend: spaCy (de_core_news_sm)
    4984 nicht-leere Dokumente nach der Bereinigung.

[3] Vektorisierung – Vergleich:
    Bag-of-Words : Vokabular=2000, Matrix=(4984, 2000), Sparsity=0.9961
    TF-IDF       : Vokabular=2000, Matrix=(4984, 2000), Sparsity=0.9961
    Top-Terme BoW   : fahren, geben, neu, zug, berlin, bus, ice, gdl, stehen, ticket
    Top-Terme TF-IDF: fahren, zug, neu, geben, berlin, bus, ice, gdl, stehen, streik

[4] Extrahierte Themen:

### LDA (auf Bag-of-Words)
Thema 1: ticket, schön, vieler, euro, einfach, sitzen, fahren, geben, bitte, bahncard, gerne, band
Thema 2: neu, planen, app, versprechen, bahnhof, deutlich, chef, service, frankfurt, zug, grube, zahl
Thema 3: fahren, zug, ice, strecke, minute, münchen, stehen, züge, geben, verspätung, hamburg, hannover
Thema 4: bus, auto, fahren, berlin, neu, weg, mann, frau, sehen, geben, zug, zeitung
Thema 5: gdl, streik, berlin, beenden, geben, schlichtung, tarifkonflikt, stehen, neu, gewerkschaft, wichtig, anderer
Thema 6: fahren, min, berlin, spät, verspätung, züge, denken, nächster, streik, köln, vieler, minute

### NMF (auf TF-IDF)
Thema 1: fahren, auto, hallo, nehmen, zug, schnell, züge, normal, berlin, samstag, nächster, genau
Thema 2: streik, gdl, beenden, schlichtung, tarifkonflikt, bahnstreik, gewerkschaft, weselsky, lokführergewerkschaft, ramelow, lokführer, lokführ
Thema 3: geben, stehen, ice, ticket, sehen, sagen, strecke, münchen, vieler, hallo, bitte, schön
Thema 4: zug, verspätung, minute, versprechen, min, chef, grube, service, umgekehrt, wagenreihung, pünktlich, nächster
Thema 5: neu, berlin, meldung, verkehrsmeldungen, leipzig, alt, hauptbahnhof, züge, halle, eröffnen, planen, sperren
Thema 6: bus, auto, kosten, kostenlos, unterwegs, hause, wlan, fernbus, verbindung, bild, ort, berlin

### LSA / TruncatedSVD (auf TF-IDF) – Bonusvergleich
Thema 1: fahren, zug, bus, berlin, geben, streik, neu, gdl, ice, stehen, ticket, verspätung
Thema 2: streik, gdl, beenden, schlichtung, tarifkonflikt, bahnstreik, gewerkschaft, berlin, weselsky, lokführergewerkschaft, ramelow, lokführer
Thema 3: fahren, streik, gdl, beenden, schlichtung, bahnstreik, tarifkonflikt, lokführergewerkschaft, lokführ, streikende, einigung, weselsky
Thema 4: neu, berlin, bus, geben, alt, meldung, auto, verkehrsmeldungen, fahren, leipzig, euro, finden
Thema 5: bus, geben, sitzen, ticket, auto, stehen, schön, verspätung, sagen, bitte, finden, einfach
Thema 6: verspätung, minute, neu, ice, min, berlin, nächster, richtung, fahren, warten, hbf, haltestelle

[5] Abbildungen gespeichert:
    - results\01_vergleich_vektorisierung.png
    - results\02_wortwolken_lda.png
    - results\02_wortwolken_nmf.png
    - results\03_themenverteilung.png

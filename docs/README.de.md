[English](../README.md) | [Chinese](README.zh.md) | [Korean](README.ko.md) | [Japanese](README.ja.md) | [German](README.de.md) | [French](README.fr.md) | [Spanish](README.es.md) | [Arabic](README.ar.md) | [Italian](README.it.md)

# Aktienkursprognosemodell

Dieses Projekt ist ein Deep-Learning-/Maschinenlernmodell, das die Aktienkurse der Top-30-KOSPI-Unternehmen mithilfe von LSTM und XGBoost vorhersagt, jetzt mit einer interaktiven Electron-basierten Desktop-Anwendung.

## 📜 Projektübersicht

Dieses Projekt zielt darauf ab, zukünftige Aktienkurse durch das Lernen aus historischen Aktiendaten vorherzusagen. Es prognostiziert speziell den Schlusskurs für ein bestimmtes Datum und bietet eine intuitive Desktop-Oberfläche für die Interaktion.

## ✨ Hauptmerkmale

  * **Interaktive Desktop-Anwendung**: Eine benutzerfreundliche Electron-basierte Benutzeroberfläche zur Verwaltung und Interaktion mit dem Aktienprognosemodell.
  * **Datenverarbeitung**: Führen Sie Skripte zur Datenvorverarbeitung direkt über die Benutzeroberfläche aus.
  * **Modellvorhersage**: Führen Sie Aktienkursprognosemodelle aus und zeigen Sie die Ergebnisse innerhalb der Anwendung an.
  * **Finanzberichte & Index-Viewer**: Durchsuchen und Anzeigen von Finanzberichten und verschiedenen Finanzindizes.
  * **Datenerfassung**: Sammelt Aktiendaten von [Datenquelle, z.B. Yahoo Finance, Naver Finance].
  * **Datenvorverarbeitung**: Verarbeitet Daten in ein geeignetes Format für die Aktienkursprognose, einschließlich gleitender Durchschnitte und Normalisierung.
  * **Modelltraining**: Trainiert Aktienkursprognosemodelle mithilfe von [Deep Learning/Maschinenlernbibliothek, z.B. TensorFlow, PyTorch, Scikit-learn].
  * **Visualisierung der Vorhersageergebnisse**: Visualisiert Diagramme, die tatsächliche und vorhergesagte Aktienkurse vergleichen, mithilfe von Matplotlib, Plotly usw.
  * **Leistungsbewertung**: Bewertet die Modellleistung mithilfe verschiedener Metriken wie MSE, RMSE und MAE.

## 🛠️ Tech Stack

  * **Desktop-Anwendung**: `Electron`, `Node.js`, `HTML`, `CSS`, `JavaScript`
  * **Sprache**: `Python 3.x`
  * **Python-Bibliotheken**:
      * `Pandas`: Datenanalyse und -manipulation
      * `NumPy`: Numerische Operationen
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: Modelle für maschinelles Lernen und Deep Learning
      * `Matplotlib` / `Seaborn` / `Plotly`: Datenvisualisierung
      * `Jupyter Notebook`: Projektentwicklungsumgebung

## 📁 Projektstruktur

```
StockPricePredictionModel/
├── .git/
├── src/
│   ├── main/
│   │   ├── IpcHandler.js
│   │   └── preload.js
│   ├── renderer/
│   │   ├── core/
│   │   │   └── ScriptRunner.js
│   │   └── ui/
│   │       ├── FileViewer.js
│   │       └── UIManager.js
│   └── styles/
│       ├── base.css
│       ├── components.css
│       ├── layout.css
│       └── navigation.css
├── Financial_Index/
├── Financial_Research/
├── LSTM/
├── Report/
├── processed/
├── raw/
├── index.html
├── main.js
├── package.json
├── renderer.js
├── style.css
├── .gitignore
├── LICENSE
├── package-lock.json
├── README.md
├── README(KOR).md
└── requirements.txt
```

## 💾 Installation & Nutzung

1.  **Git-Repository klonen:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **Python-Abhängigkeiten installieren:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Node.js-Abhängigkeiten installieren:**

    ```bash
    npm install
    ```

4.  **Desktop-Anwendung starten:**

    ```bash
    npm start
    ```

    Alternativ können Sie Jupyter Notebook weiterhin für Entwicklung und Experimente verwenden:

    ```bash
    jupyter notebook
    ```

    *   Öffnen Sie `[Dateiname für Datenerfassung und Vorverarbeitung].ipynb` und führen Sie die Zellen nacheinander aus, um die Daten vorzubereiten.
    *   Öffnen Sie `[Dateiname für Modelltraining und -bewertung].ipynb`, um das Modell zu trainieren und die Ergebnisse zu überprüfen.

## 📊 Datensatz

  * **Datenquelle**: [Datenquelle, z.B. Yahoo Finance API, Naver Finance Crawling]
  * **Ziel**: Top 30 KOSPI-Aktien
  * **Zeitraum**: [Datenzeitraum, z.B. 2010-01-01 \~ 2023-12-31]
  * **Verwendete Merkmale**: `Open`, `High`, `Low`, `Close`, `Volume`, etc.

**Modellleistung:**

  * **LN**: 2~3

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details finden Sie in der Datei `LICENSE`.

## Mitwirkende

  - Gominseo, Kim Jiho, Kim Taeun, Lee Humin, Lee Jaehwan

[English](../README.md) | [Chinese](README.zh.md) | [Korean](README.ko.md) | [Japanese](README.ja.md) | [German](README.de.md) | [French](README.fr.md) | [Spanish](README.es.md) | [Arabic](README.ar.md) | [Italian](README.it.md)

# 株価予測モデル

このプロジェクトは、LSTMとXGBoostを使用してKOSPI上位30社の株価を予測する深層学習/機械学習モデルであり、インタラクティブなElectronベースのデスクトップアプリケーションとして提供されます。

## 📜 プロジェクト概要

このプロジェクトは、過去の株価データから学習することで将来の株価を予測することを目的としています。特定の日付の終値を予測し、直感的なデスクトップインターフェースを提供します。

## ✨ 主要機能

*   **インタラクティブなデスクトップアプリケーション**: 株価予測モデルを管理・操作するための、ユーザーフレンドリーなElectronベースのUI。
*   **データ処理**: UIから直接データ前処理スクリプトを実行。
*   **モデル予測**: 株価予測モデルを実行し、アプリケーション内で結果を表示。
*   **財務報告書・インデックスビューア**: 財務報告書や様々な金融インデックスを閲覧・表示。
*   **データ収集**: [データソース、例：Yahoo Finance, Naver Finance]から株価データを収集。
*   **データ前処理**: 移動平均や正規化など、株価予測に適した形式にデータを処理。
*   **モデル学習**: [使用するディープラーニング/機械学習ライブラリ、例：TensorFlow, PyTorch, Scikit-learn]を使用して株価予測モデルを学習。
*   **予測結果の可視化**: Matplotlib、Plotlyなどを使用して、実際の株価と予測された株価を比較するグラフを可視化。
*   **性能評価**: MSE、RMSE、MAEなどの様々な指標を使用してモデルの性能を評価。

## 🛠️ 技術スタック

  * **デスクトップアプリケーション**: `Electron`, `Node.js`, `HTML`, `CSS`, `JavaScript`
  * **言語**: `Python 3.x`
  * **Python ライブラリ**:
      * `Pandas`: データ分析と操作
      * `NumPy`: 数値演算
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: 機械学習および深層学習モデル
      * `Matplotlib` / `Seaborn` / `Plotly`: データ可視化
      * `Jupyter Notebook`: プロジェクト開発環境

## 📁 プロジェクト構成

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

## 💾 インストールと使用方法

1.  **Gitリポジトリをクローンする:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **Pythonの依存関係をインストールする:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Node.jsの依存関係をインストールする:**

    ```bash
    npm install
    ```

4.  **デスクトップアプリケーションを実行する:**

    ```bash
    npm start
    ```

    または、開発と実験のためにJupyter Notebookを使用することもできます。

    ```bash
    jupyter notebook
    ```

    *   `[Data Collection and Preprocessing Filename].ipynb` を開き、セルを順番に実行してデータを準備します。
    *   `[Model Training and Evaluation Filename].ipynb` を開いてモデルをトレーニングし、結果を確認します。

## 📊 データセット

  * **データソース**: [データソース, 例: Yahoo Finance API, Naver Finance クローリング]
  * **ターゲット**: KOSPI上位30銘柄
  * **期間**: [データ期間, 例: 2010-01-01 ～ 2023-12-31]
  * **使用された特徴量**: `始値`, `高値`, `安値`, `終値`, `出来高`, など

**モデル性能:**

  * **LN**: 2~3

## 📄 ライセンス

このプロジェクトはMITライセンスの下で提供されています。詳細については、`LICENSE`ファイルをご覧ください。

## 貢献者

- Gominseo、Kim Jiho、Kim Taeun、Lee Humin、Lee Jaehwan

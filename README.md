[English](README.md) | [Chinese](docs/README.zh.md) | [Korean](docs/README.ko.md) | [Japanese](docs/README.ja.md) | [German](docs/README.de.md) | [French](docs/README.fr.md) | [Spanish](docs/README.es.md) | [Arabic](docs/README.ar.md) | [Italian](docs/README.it.md)

# Stock Price Prediction Model

This project is a deep learning/machine learning model that predicts the stock prices of the top 30 KOSPI companies using LSTM and XGBoost, now with an interactive Electron-based desktop application.

## 📜 Project Overview

This project aims to predict future stock prices by learning from historical stock data. It specifically predicts the closing price for a given date, offering an intuitive desktop interface for interaction.

## ✨ Key Features

  * **Interactive Desktop Application**: A user-friendly Electron-based UI to manage and interact with the stock prediction model.
  * **Data Processing**: Run data preprocessing scripts directly from the UI.
  * **Model Prediction**: Execute stock price prediction models and view results within the application.
  * **Financial Reports & Indexes Viewer**: Browse and display financial reports and various financial indexes.
  * **Data Collection**: Collects stock data from [Data Source, e.g., Yahoo Finance, Naver Finance].
  * **Data Preprocessing**: Processes data into a suitable format for stock price prediction, including moving averages and normalization.
  * **Model Training**: Trains stock price prediction models using [Deep Learning/Machine Learning Library Used, e.g., TensorFlow, PyTorch, Scikit-learn].
  * **Prediction Result Visualization**: Visualizes graphs comparing actual and predicted stock prices using Matplotlib, Plotly, etc.
  * **Performance Evaluation**: Evaluates model performance using various metrics such as MSE, RMSE, and MAE.

## 🛠️ Tech Stack

  * **Desktop Application**: `Electron`, `Node.js`, `HTML`, `CSS`, `JavaScript`
  * **Language**: `Python 3.x`
  * **Python Libraries**:
      * `Pandas`: Data analysis and manipulation
      * `NumPy`: Numerical operations
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: Machine learning and deep learning models
      * `Matplotlib` / `Seaborn` / `Plotly`: Data visualization
      * `Jupyter Notebook`: Project development environment

## 📁 Project Structure

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

## 💾 Installation & Usage

1.  **Clone the Git Repository:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Node.js Dependencies:**

    ```bash
    npm install
    ```

4.  **Run the Desktop Application:**

    ```bash
    npm start
    ```

    Alternatively, you can still use Jupyter Notebook for development and experimentation:

    ```bash
    jupyter notebook
    ```

    *   Open `[Data Collection and Preprocessing Filename].ipynb` and run the cells sequentially to prepare the data.
    *   Open `[Model Training and Evaluation Filename].ipynb` to train the model and check the results.

## 📊 Dataset

  * **Data Source**: [Data Source, e.g., Yahoo Finance API, Naver Finance crawling]
  * **Target**: Top 30 KOSPI stocks
  * **Period**: [Data Period, e.g., 2010-01-01 \~ 2023-12-31]
  * **Features Used**: `Open`, `High`, `Low`, `Close`, `Volume`, etc.

**Model Performance:**

  * **LN**: 2~3

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributors

  - Gominseo, Kim Jiho, Kim Taeun, Lee Humin, Lee Jaehwan
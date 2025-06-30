# Stock Price Prediction Model

This project is a deep learning/machine learning model that predicts the stock prices of the top 30 KOSPI companies using LSTM and XGBoost.

## 📜 Project Overview

This project aims to predict future stock prices by learning from historical stock data. It specifically predicts the closing price for a given date.

## ✨ Key Features

  * **Data Collection**: Collects stock data from [Data Source, e.g., Yahoo Finance, Naver Finance].
  * **Data Preprocessing**: Processes data into a suitable format for stock price prediction, including moving averages and normalization.
  * **Model Training**: Trains stock price prediction models using [Deep Learning/Machine Learning Library Used, e.g., TensorFlow, PyTorch, Scikit-learn].
  * **Prediction Result Visualization**: Visualizes graphs comparing actual and predicted stock prices using Matplotlib, Plotly, etc.
  * **Performance Evaluation**: Evaluates model performance using various metrics such as MSE, RMSE, and MAE.

## 🛠️ Tech Stack

  * **Language**: `Python 3.x`
  * **Libraries**:
      * `Pandas`: Data analysis and manipulation
      * `NumPy`: Numerical operations
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: Machine learning and deep learning models
      * `Matplotlib` / `Seaborn` / `Plotly`: Data visualization
      * `Jupyter Notebook`: Project development environment

## 💾 Installation & Usage

1.  **Clone the Git Repository:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **Install Required Libraries:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

4.  **Execute Notebook Files:**

      * Open `[Data Collection and Preprocessing Filename].ipynb` and run the cells sequentially to prepare the data.
      * Open `[Model Training and Evaluation Filename].ipynb` to train the model and check the results.

## 📊 Dataset

  * **Data Source**: [Data Source, e.g., Yahoo Finance API, Naver Finance crawling]
  * **Target**: Top 30 KOSPI stocks
  * **Period**: [Data Period, e.g., 2010-01-01 \~ 2023-12-31]
  * **Features Used**: `Open`, `High`, `Low`, `Close`, `Volume`, etc.

**Model Performance:**

  * **LN**: 2~3

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

-----

**Usage Notes:**

  * Save the content above as `Readme.md` and add it to your GitHub repository.
  * The content within `[ ]` and examples should be modified to fit your project.
  * If you don't have a `requirements.txt` file, you can generate one using the command: `pip freeze > requirements.txt`
  * For prediction result images, it's convenient to upload them to GitHub Issues or other platforms and then use the image links.

## Contributors

  - Gominseo, Kim Jiho, Kim Taeun, Lee Humin, Lee Jaehwan

-----
[English](../README.md) | [Chinese](README.zh.md) | [Korean](README.ko.md) | [Japanese](README.ja.md) | [German](README.de.md) | [French](README.fr.md) | [Spanish](README.es.md) | [Arabic](README.ar.md) | [Italian](README.it.md)

# 股票价格预测模型

本项目是一个深度学习/机器学习模型，使用 LSTM 和 XGBoost 预测 KOSPI 前 30 家公司的股票价格，现在附带一个交互式 Electron 桌面应用程序。

## 📜 项目概览

本项目旨在通过学习历史股票数据来预测未来的股票价格。它专门预测给定日期的收盘价，并提供一个直观的桌面界面进行交互。

## ✨ 主要功能

  * **交互式桌面应用程序**：一个用户友好的基于 Electron 的 UI，用于管理和与股票预测模型进行交互。
  * **数据处理**：直接从 UI 运行数据预处理脚本。
  * **模型预测**：执行股票价格预测模型并在应用程序内查看结果。
  * **财务报告与指数查看器**：浏览和显示财务报告以及各种金融指数。
  * **数据收集**：从 [数据源，例如：Yahoo Finance, Naver Finance] 收集股票数据。
  * **数据预处理**：将数据处理成适合股票价格预测的格式，包括移动平均线和归一化。
  * **模型训练**：使用 [深度学习/机器学习库，例如：TensorFlow, PyTorch, Scikit-learn] 训练股票价格预测模型。
  * **预测结果可视化**：使用 Matplotlib、Plotly 等工具可视化实际和预测股票价格的图表。
  * **性能评估**：使用 MSE、RMSE 和 MAE 等各种指标评估模型性能。

## 🛠️ 技术栈

  * **桌面应用程序**：`Electron`, `Node.js`, `HTML`, `CSS`, `JavaScript`
  * **语言**：`Python 3.x`
  * **Python 库**：
      * `Pandas`：数据分析和处理
      * `NumPy`：数值运算
      * `TensorFlow` / `PyTorch` / `Scikit-learn`：机器学习和深度学习模型
      * `Matplotlib` / `Seaborn` / `Plotly`：数据可视化
      * `Jupyter Notebook`：项目开发环境

## 📁 项目结构

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

## 💾 安装与使用

1.  **克隆 Git 仓库：**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **安装 Python 依赖：**

    ```bash
    pip install -r requirements.txt
    ```

3.  **安装 Node.js 依赖：**

    ```bash
    npm install
    ```

4.  **运行桌面应用程序：**

    ```bash
    npm start
    ```

    或者，您仍然可以使用 Jupyter Notebook 进行开发和实验：

    ```bash
    jupyter notebook
    ```

    *   打开 `[数据收集和预处理文件名].ipynb` 并按顺序运行单元格以准备数据。
    *   打开 `[模型训练和评估文件名].ipynb` 以训练模型并检查结果。

## 📊 数据集

  * **数据源**：[数据源，例如：Yahoo Finance API, Naver Finance 爬取]
  * **目标**：KOSPI 前 30 支股票
  * **时间段**：[数据时间段，例如：2010-01-01 \~ 2023-12-31]
  * **使用的特征**：`Open`（开盘价）, `High`（最高价）, `Low`（最低价）, `Close`（收盘价）, `Volume`（成交量）等。

**模型性能：**

  * **LN**：2~3

## 📄 许可证

本项目采用 MIT 许可证。更多详情请参阅 `LICENSE` 文件。

## 贡献者

  - Gominseo, Kim Jiho, Kim Taeun, Lee Humin, Lee Jaehwan

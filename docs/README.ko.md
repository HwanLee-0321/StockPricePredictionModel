[English](../README.md) | [Chinese](README.zh.md) | [Korean](README.ko.md) | [Japanese](README.ja.md) | [German](README.de.md) | [French](README.fr.md) | [Spanish](README.es.md) | [Arabic](README.ar.md) | [Italian](README.it.md)

# 주식 가격 예측 모델

이 프로젝트는 LSTM과 XGBoost를 사용하여 KOSPI 상위 30개 기업의 주식 가격을 예측하는 딥러닝/머신러닝 모델이며, 이제 인터랙티브한 Electron 기반 데스크톱 애플리케이션을 제공합니다.

## 📜 프로젝트 개요

이 프로젝트는 과거 주식 데이터를 학습하여 미래 주식 가격을 예측하는 것을 목표로 합니다. 특정 날짜의 종가를 예측하며, 직관적인 데스크톱 인터페이스를 통해 상호작용을 제공합니다.

## ✨ 주요 기능

  * **인터랙티브 데스크톱 애플리케이션**: 주식 예측 모델을 관리하고 상호작용하기 위한 사용자 친화적인 Electron 기반 UI.
  * **데이터 처리**: UI에서 직접 데이터 전처리 스크립트를 실행합니다.
  * **모델 예측**: 주식 가격 예측 모델을 실행하고 애플리케이션 내에서 결과를 확인합니다.
  * **재무 보고서 및 지수 뷰어**: 재무 보고서와 다양한 금융 지수를 탐색하고 표시합니다.
  * **데이터 수집**: [데이터 소스, 예: Yahoo Finance, Naver Finance]에서 주식 데이터를 수집합니다.
  * **데이터 전처리**: 이동 평균 및 정규화를 포함하여 주식 가격 예측에 적합한 형식으로 데이터를 처리합니다.
  * **모델 훈련**: [사용된 딥러닝/머신러닝 라이브러리, 예: TensorFlow, PyTorch, Scikit-learn]를 사용하여 주식 가격 예측 모델을 훈련합니다.
  * **예측 결과 시각화**: Matplotlib, Plotly 등을 사용하여 실제 주식 가격과 예측 주식 가격을 비교하는 그래프를 시각화합니다.
  * **성능 평가**: MSE, RMSE, MAE와 같은 다양한 지표를 사용하여 모델 성능을 평가합니다.

## 🛠️ 기술 스택

  * **데스크톱 애플리케이션**: `Electron`, `Node.js`, `HTML`, `CSS`, `JavaScript`
  * **언어**: `Python 3.x`
  * **Python 라이브러리**:
      * `Pandas`: 데이터 분석 및 조작
      * `NumPy`: 수치 연산
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: 머신러닝 및 딥러닝 모델
      * `Matplotlib` / `Seaborn` / `Plotly`: 데이터 시각화
      * `Jupyter Notebook`: 프로젝트 개발 환경

## 📁 프로젝트 구조

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

## 💾 설치 및 사용법

1.  **Git 저장소 복제:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **Python 종속성 설치:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Node.js 종속성 설치:**

    ```bash
    npm install
    ```

4.  **데스크톱 애플리케이션 실행:**

    ```bash
    npm start
    ```

    또는 Jupyter Notebook을 사용하여 개발 및 실험을 계속할 수 있습니다:

    ```bash
    jupyter notebook
    ```

    *   `[데이터 수집 및 전처리 파일명].ipynb`을 열고 셀을 순차적으로 실행하여 데이터를 준비합니다.
    *   `[모델 훈련 및 평가 파일명].ipynb`을 열어 모델을 훈련하고 결과를 확인합니다.

## 📊 데이터셋

  * **데이터 소스**: [데이터 소스, 예: Yahoo Finance API, Naver Finance 크롤링]
  * **대상**: KOSPI 상위 30개 주식
  * **기간**: [데이터 기간, 예: 2010-01-01 ~ 2023-12-31]
  * **사용된 특징**: `Open`, `High`, `Low`, `Close`, `Volume` 등

**모델 성능:**

  * **LN**: 2~3

## 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 `LICENSE` 파일을 참조하십시오.

## 기여자

  - Gominseo, Kim Jiho, Kim Taeun, Lee Humin, Lee Jaehwan

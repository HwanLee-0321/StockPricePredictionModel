# 주가 예측 모델 (Stock Price Prediction Model)

이 프로젝트는 LSTM과 XGBoost을 이용하여 코스피 상위 30위의 주가를 예측하는 딥러닝/머신러닝 모델입니다.

## 📜 프로젝트 개요 (Overview)

본 프로젝트는 과거 주가 데이터를 학습하여 미래의 주가를 예측하는 것을 목표로 합니다. 특정 날짜의 종가를 예측합니다.

## ✨ 주요 기능 (Features)

  * **데이터 수집**: [데이터 출처, 예: Yahoo Finance, 네이버 금융]에서 주가 데이터를 수집합니다.
  * **데이터 전처리**: 이동 평균, 정규화 등 주가 예측에 적합한 형태로 데이터를 가공합니다.
  * **모델 학습**: [사용한 딥러닝/머신러닝 라이브러리, 예: TensorFlow, PyTorch, Scikit-learn]를 사용하여 주가 예측 모델을 학습합니다.
  * **예측 결과 시각화**: Matplotlib, Plotly 등을 사용하여 실제 주가와 예측 주가를 비교하는 그래프를 시각화합니다.
  * **성능 평가**: MSE, RMSE, MAE 등 다양한 지표를 통해 모델의 성능을 평가합니다.

## 🛠️ 기술 스택 (Tech Stack)

  * **Language**: `Python 3.x`
  * **Libraries**:
      * `Pandas`: 데이터 분석 및 처리
      * `NumPy`: 수치 연산
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: 머신러닝 및 딥러닝 모델
      * `Matplotlib` / `Seaborn` / `Plotly`: 데이터 시각화
      * `Jupyter Notebook`: 프로젝트 개발 환경

## 💾 설치 및 실행 (Installation & Usage)

1.  **Git 리포지토리 클론:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **필요한 라이브러리 설치:**

    ```bash
    pip install -r requirements.txt
    ```
  
3.  **Jupyter Notebook 실행:**

    ```bash
    jupyter notebook
    ```

4.  **노트북 파일 실행:**

      * `[데이터 수집 및 전처리 파일명].ipynb` 파일을 열어 순서대로 셀을 실행하여 데이터를 준비합니다.
      * `[모델 학습 및 평가 파일명].ipynb` 파일을 열어 모델을 학습하고 결과를 확인합니다.

## 📊 데이터셋 (Dataset)

  * **데이터 출처**: [데이터 출처, 예: Yahoo Finance API, 네이버 금융 크롤링]
  * **대상**: 코스피 상위 30개의 종목
  * **기간**: [데이터 기간, 예: 2010-01-01 \~ 2023-12-31]
  * **사용한 피처**: `Open`, `High`, `Low`, `Close`, `Volume` 등

**모델 성능:**

  * **LN**: 2~3

## 📄 라이선스 (License)

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참고하세요.

-----

**사용 안내:**

  * 위 내용을 `Readme.md` 파일로 저장하여 GitHub 리포지토리에 추가하세요.
  * `[ ]` 안에 있는 내용과 예시는 프로젝트에 맞게 수정해야 합니다.
  * `requirements.txt` 파일이 없다면, 다음 명령어를 통해 생성할 수 있습니다: `pip freeze > requirements.txt`
  * 예측 결과 이미지는 GitHub 이슈나 다른 곳에 업로드한 후 이미지 링크를 가져와서 사용하면 편리합니다.

## 기여자목록

- 고민서, 김지호, 김태은, 이후민, 이재환 
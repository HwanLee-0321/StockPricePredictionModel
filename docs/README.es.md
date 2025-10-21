[English](../README.md) | [Chinese](README.zh.md) | [Korean](README.ko.md) | [Japanese](README.ja.md) | [German](README.de.md) | [French](README.fr.md) | [Spanish](README.es.md) | [Arabic](README.ar.md) | [Italian](README.it.md)

# Modelo de Predicción del Precio de Acciones

Este proyecto es un modelo de aprendizaje profundo/aprendizaje automático que predice los precios de las acciones de las 30 principales empresas KOSPI utilizando LSTM y XGBoost, ahora con una aplicación de escritorio interactiva basada en Electron.

## 📜 Resumen del Proyecto

Este proyecto tiene como objetivo predecir los precios futuros de las acciones aprendiendo de datos históricos de las mismas. Específicamente, predice el precio de cierre para una fecha determinada, ofreciendo una interfaz de escritorio intuitiva para la interacción.

## ✨ Características Clave

*   **Aplicación de Escritorio Interactiva**: Una interfaz de usuario amigable basada en Electron para gestionar e interactuar con el modelo de predicción de acciones.
*   **Procesamiento de Datos**: Ejecuta scripts de preprocesamiento de datos directamente desde la interfaz de usuario.
*   **Predicción del Modelo**: Ejecuta modelos de predicción de precios de acciones y visualiza los resultados dentro de la aplicación.
*   **Visualizador de Informes Financieros e Índices**: Explora y muestra informes financieros y varios índices financieros.
*   **Recopilación de Datos**: Recopila datos de acciones de [Fuente de Datos, ej., Yahoo Finance, Naver Finance].
*   **Preprocesamiento de Datos**: Procesa los datos a un formato adecuado para la predicción del precio de las acciones, incluyendo medias móviles y normalización.
*   **Entrenamiento del Modelo**: Entrena modelos de predicción de precios de acciones utilizando [Biblioteca de Deep Learning/Machine Learning utilizada, ej., TensorFlow, PyTorch, Scikit-learn].
*   **Visualización de Resultados de Predicción**: Visualiza gráficos que comparan los precios de acciones reales y predichos utilizando Matplotlib, Plotly, etc.
*   **Evaluación del Rendimiento**: Evalúa el rendimiento del modelo utilizando varias métricas como MSE, RMSE y MAE.

## 🛠️ Pila Tecnológica

  * **Aplicación de Escritorio**: `Electron`, `Node.js`, `HTML`, `CSS`, `JavaScript`
  * **Lenguaje**: `Python 3.x`
  * **Librerías de Python**:
      * `Pandas`: Análisis y manipulación de datos
      * `NumPy`: Operaciones numéricas
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: Modelos de aprendizaje automático y aprendizaje profundo
      * `Matplotlib` / `Seaborn` / `Plotly`: Visualización de datos
      * `Jupyter Notebook`: Entorno de desarrollo de proyectos

## 📁 Estructura del Proyecto

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

## 💾 Instalación y Uso

1.  **Clona el Repositorio de Git:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **Instala las Dependencias de Python:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Instala las Dependencias de Node.js:**

    ```bash
    npm install
    ```

4.  **Ejecuta la Aplicación de Escritorio:**

    ```bash
    npm start
    ```

    Alternativamente, aún puedes usar Jupyter Notebook para desarrollo y experimentación:

    ```bash
    jupyter notebook
    ```

    *   Abre `[Nombre del archivo de Recopilación y Preprocesamiento de Datos].ipynb` y ejecuta las celdas secuencialmente para preparar los datos.
    *   Abre `[Nombre del archivo de Entrenamiento y Evaluación del Modelo].ipynb` para entrenar el modelo y verificar los resultados.

## 📊 Conjunto de datos

  * **Fuente de datos**: [Fuente de datos, p. ej., API de Yahoo Finance, rastreo de Naver Finance]
  * **Objetivo**: Las 30 principales acciones de KOSPI
  * **Período**: [Período de datos, p. ej., 01-01-2010 ~ 31-12-2023]
  * **Características utilizadas**: `Apertura`, `Máximo`, `Mínimo`, `Cierre`, `Volumen`, etc.

**Rendimiento del modelo:**

  * **LN**: 2~3

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Contribuidores

  - Gominseo, Kim Jiho, Kim Taeun, Lee Humin, Lee Jaehwan

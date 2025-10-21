[English](../README.md) | [Chinese](README.zh.md) | [Korean](README.ko.md) | [Japanese](README.ja.md) | [German](README.de.md) | [French](README.fr.md) | [Spanish](README.es.md) | [Arabic](README.ar.md) | [Italian](README.it.md)

# نموذج التنبؤ بأسعار الأسهم

هذا المشروع عبارة عن نموذج تعلم عميق/تعلم آلي يتنبأ بأسعار أسهم أفضل 30 شركة في مؤشر KOSPI باستخدام LSTM و XGBoost، والآن مع تطبيق سطح مكتب تفاعلي قائم على Electron.

## 📜 نظرة عامة على المشروع

يهدف هذا المشروع إلى التنبؤ بأسعار الأسهم المستقبلية من خلال التعلم من بيانات الأسهم التاريخية. يتنبأ على وجه التحديد بسعر الإغلاق لتاريخ معين، ويوفر واجهة سطح مكتب بديهية للتفاعل.

## ✨ الميزات الرئيسية

  * **تطبيق سطح مكتب تفاعلي**: واجهة مستخدم سهلة الاستخدام قائمة على Electron لإدارة نموذج التنبؤ بالأسهم والتفاعل معه.
  * **معالجة البيانات**: تشغيل نصوص معالجة البيانات المسبقة مباشرة من واجهة المستخدم.
  * **تنبؤ النموذج**: تنفيذ نماذج التنبؤ بأسعار الأسهم وعرض النتائج داخل التطبيق.
  * **عارض التقارير المالية والمؤشرات**: تصفح وعرض التقارير المالية والمؤشرات المالية المختلفة.
  * **جمع البيانات**: يجمع بيانات الأسهم من [مصدر البيانات، على سبيل المثال، Yahoo Finance، Naver Finance].
  * **المعالجة المسبقة للبيانات**: يعالج البيانات إلى تنسيق مناسب للتنبؤ بأسعار الأسهم، بما في ذلك المتوسطات المتحركة والتطبيع.
  * **تدريب النموذج**: يدرب نماذج التنبؤ بأسعار الأسهم باستخدام [مكتبة التعلم العميق/التعلم الآلي المستخدمة، على سبيل المثال، TensorFlow، PyTorch، Scikit-learn].
  * **تصور نتائج التنبؤ**: يعرض رسومًا بيانية تقارن أسعار الأسهم الفعلية والمتوقعة باستخدام Matplotlib، Plotly، إلخ.
  * **تقييم الأداء**: يقيم أداء النموذج باستخدام مقاييس مختلفة مثل MSE و RMSE و MAE.

## 🛠️ حزمة التقنيات

  * **تطبيق سطح المكتب**: `Electron`, `Node.js`, `HTML`, `CSS`, `JavaScript`
  * **اللغة**: `Python 3.x`
  * **مكتبات بايثون**:
      * `Pandas`: تحليل البيانات ومعالجتها
      * `NumPy`: العمليات العددية
      * `TensorFlow` / `PyTorch` / `Scikit-learn`: نماذج التعلم الآلي والتعلم العميق
      * `Matplotlib` / `Seaborn` / `Plotly`: تصور البيانات
      * `Jupyter Notebook`: بيئة تطوير المشروع

## 📁 هيكل المشروع

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

## 💾 التثبيت والاستخدام

1.  **استنساخ مستودع Git:**

    ```bash
    git clone https://github.com/HwanLee-0321/StockPricePredictionModel.git
    cd StockPricePredictionModel
    ```

2.  **تثبيت تبعيات بايثون:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **تثبيت تبعيات Node.js:**

    ```bash
    npm install
    ```

4.  **تشغيل تطبيق سطح المكتب:**

    ```bash
    npm start
    ```

    بدلاً من ذلك، لا يزال بإمكانك استخدام Jupyter Notebook للتطوير والتجريب:

    ```bash
    jupyter notebook
    ```

    *   افتح `[اسم ملف جمع البيانات والمعالجة المسبقة].ipynb` وقم بتشغيل الخلايا بالتسلسل لإعداد البيانات.
    *   افتح `[اسم ملف تدريب النموذج وتقييمه].ipynb` لتدريب النموذج والتحقق من النتائج.

## 📊 مجموعة البيانات

  * **مصدر البيانات**: [مصدر البيانات، على سبيل المثال، Yahoo Finance API، Naver Finance crawling]
  * **الهدف**: أفضل 30 سهمًا في مؤشر KOSPI
  * **الفترة**: [فترة البيانات، على سبيل المثال، 2010-01-01 ~ 2023-12-31]
  * **الميزات المستخدمة**: `Open`, `High`, `Low`, `Close`, `Volume`, إلخ.

**أداء النموذج:**

  * **LN**: 2~3

## 📄 الترخيص

هذا المشروع مرخص بموجب ترخيص MIT. راجع ملف `LICENSE` لمزيد من التفاصيل.

## المساهمون

  - Gominseo, Kim Jiho, Kim Taeun, Lee Humin, Lee Jaehwan

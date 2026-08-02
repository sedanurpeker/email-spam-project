# 🛡️ Explainable Spam Detection System (BiGRU + Attention)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6f00.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning based SMS/email spam classifier that goes beyond a plain "spam or not" label — it also explains **why** a message was flagged, through a modern, presentation-ready Streamlit interface.

🇹🇷 *Türkçe açıklama için aşağıya inebilirsiniz.*

<!-- 📸 TODO: Add a screenshot or GIF of the Streamlit app here, e.g.:
![App demo](docs/demo.gif)
-->

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#️-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Running the App Locally](#️-running-the-app-locally)
- [Limitations](#️-limitations)
- [Tech Stack](#️-tech-stack)
- [License](#-license)
- [🇹🇷 Türkçe](#-açıklanabilir-spam-tespit-sistemi-bigru--attention)

---

## 📌 Overview

This project builds and compares several machine learning and deep learning models to classify SMS messages as **Spam** or **Ham** (legitimate), using the classic SMS Spam Collection dataset (5,572 messages).

The final production model — a **Bidirectional GRU network with a custom Attention layer** — is deployed in an interactive Streamlit web app that predicts the label, shows the spam probability, a model-confidence indicator, and a rule-based, human-readable explanation of the keyword patterns that likely influenced the decision.

---

## ✨ Features

- **Full EDA pipeline** — class balance, message-length distribution, most frequent words per class
- **Text preprocessing** — lowercasing, URL/number normalization, punctuation removal
- **Classical ML baselines** — Naive Bayes, Logistic Regression, Linear SVM, Random Forest (TF-IDF features)
- **Deep learning models** — BiGRU and BiGRU + Attention (Keras/TensorFlow)
- **Model comparison** with Accuracy / Precision / Recall / F1 and ROC-AUC
- **Streamlit demo app** with:
  - Spam probability & confidence score
  - Rule-based "why was this flagged" explanation panel
  - Expandable technical view of the preprocessed model input
  - Modern glassmorphism-style custom UI

---

## 🗂️ Project Structure

```
├── notebooks/
│   └── spam_detection.ipynb      # EDA, preprocessing, model training & evaluation
├── dataset/
│   └── SMSSpamCollection          # Raw dataset (tab-separated: label, message)
├── outputs/
│   └── models/
│       ├── bigru_attention_model.keras
│       ├── tokenizer.pkl
│       └── config.pkl
├── model_results.csv              # Metrics summary for all trained models
├── app.py                         # Streamlit web application
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 📊 Dataset

| Metric | Value |
|---|---|
| Total messages | 5,572 |
| Ham | 4,825 (86.6%) |
| Spam | 747 (13.4%) |
| Avg. ham length | ~71 characters |
| Avg. spam length | ~139 characters |

---

## 🧪 Methodology

1. **EDA** — class distribution plot, message-length histogram by class, most common words per class (`Counter`)
2. **Cleaning** — lowercase → replace URLs → replace digits with `"number"` token → strip punctuation → normalize whitespace
3. **Label encoding** — ham → 0, spam → 1
4. **Split** — 80/20 stratified train/test split
5. **Classical models** — TF-IDF (5,000 features, English stop-words) + Naive Bayes / Logistic Regression / Linear SVM / Random Forest
6. **Deep learning** — Keras Tokenizer (vocab 10,000) + padded sequences (max length 100) → Embedding → Bidirectional GRU → custom Attention layer → Dense → Sigmoid

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes | 97.31% | 100.00% | 79.87% | 88.81% |
| Logistic Regression | 97.67% | 95.56% | 86.58% | 90.85% |
| Linear SVM | 98.48% | 95.21% | 93.29% | 94.24% |
| Random Forest | 98.30% | 100.00% | 87.25% | 93.19% |
| BiGRU | 98.65% | 97.86% | 91.95% | 94.81% |
| BiGRU + Attention | 98.65% | 97.86% | 91.95% | 94.81% |

> ⚠️ **Note:** BiGRU and BiGRU + Attention currently show identical metrics above — double-check `model_results.csv` and re-run evaluation for the attention variant, since this is likely a copy artifact rather than the true result.

The **BiGRU + Attention** model was selected as the final deployed model for its strong balance of precision and recall, plus the interpretability benefits of the attention mechanism.

---

## 🖥️ Running the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## ⚠️ Limitations

- Trained on English SMS text — performance on other languages or long-form emails may vary
- The "explanation" panel in the app is a **rule-based keyword heuristic**, not the model's real attention weights — it's meant as a human-readable aid, not a mechanistic interpretation
- Not designed for phishing-specific or multilingual detection out of the box

---

## 🛠️ Tech Stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Seaborn / Matplotlib · Streamlit

---

## 📄 License

This project is licensed under the **MIT License** — feel free to fork, adapt, and build on it for educational and portfolio purposes.

---
---

# 🇹🇷 Açıklanabilir Spam Tespit Sistemi (BiGRU + Attention)

Bir mesajı yalnızca "spam mı değil mi" diye etiketlemekle kalmayıp, **neden** spam olarak işaretlendiğini de açıklayan, derin öğrenme tabanlı bir SMS/e-posta spam sınıflandırıcısı. Modern ve sunuma hazır bir Streamlit arayüzü ile birlikte gelir.

<!-- 📸 TODO: Uygulamanın ekran görüntüsünü veya GIF'ini buraya ekleyin -->

## 📌 Genel Bakış

Bu proje, klasik SMS Spam Collection veri seti (5.572 mesaj) kullanılarak SMS mesajlarını **Spam** veya **Ham** (normal) olarak sınıflandırmak için çeşitli makine öğrenmesi ve derin öğrenme modellerini eğitip karşılaştırır.

Üretime alınan nihai model — özel bir Attention katmanına sahip **Bidirectional GRU (BiGRU)** ağı — etkileşimli bir Streamlit web uygulamasında yayınlanır. Uygulama; tahmini, spam olasılığını, model güven seviyesini ve kararı muhtemelen etkileyen anahtar kelime kalıplarına dayalı, kural tabanlı ve insan tarafından okunabilir bir açıklamayı gösterir.

## ✨ Özellikler

- 📊 **Kapsamlı KVA (EDA) süreci** — sınıf dengesi, mesaj uzunluğu dağılımı, sınıf başına en sık kelimeler
- 🧹 **Metin ön işleme** — küçük harfe çevirme, URL/sayı normalizasyonu, noktalama işaretlerinin temizlenmesi
- 🤖 **Klasik ML temel modelleri** — Naive Bayes, Lojistik Regresyon, Linear SVM, Random Forest (TF-IDF özellikleri)
- 🧠 **Derin öğrenme modelleri** — BiGRU ve BiGRU + Attention (Keras/TensorFlow)
- 📈 **Model karşılaştırması** — Accuracy / Precision / Recall / F1 ve ROC-AUC metrikleri
- 🖥️ **Streamlit demo uygulaması özellikleri:**
  - Spam olasılığı ve güven skoru
  - "Neden bu şekilde işaretlendi?" kural tabanlı açıklama paneli
  - Modele giden ön işlenmiş metni gösteren genişletilebilir teknik detay bölümü
  - Modern cam efektli (glassmorphism) özel arayüz

## 🗂️ Proje Yapısı

```
├── notebooks/
│   └── spam_detection.ipynb      # KVA, ön işleme, model eğitimi ve değerlendirme
├── dataset/
│   └── SMSSpamCollection          # Ham veri seti (tab ile ayrılmış: etiket, mesaj)
├── outputs/
│   └── models/
│       ├── bigru_attention_model.keras
│       ├── tokenizer.pkl
│       └── config.pkl
├── model_results.csv              # Tüm eğitilen modellerin metrik özeti
├── app.py                         # Streamlit web uygulaması
├── requirements.txt               # Proje bağımlılıkları
└── README.md
```

## 📊 Veri Seti

| Metrik | Değer |
|---|---|
| Toplam mesaj | 5.572 |
| Ham (normal) | 4.825 (%86,6) |
| Spam | 747 (%13,4) |
| Ort. ham uzunluğu | ~71 karakter |
| Ort. spam uzunluğu | ~139 karakter |

## 🧪 Yöntem

1. **KVA** — sınıf dağılımı grafiği, sınıfa göre mesaj uzunluğu histogramı, sınıf başına en sık kelimeler (`Counter`)
2. **Temizleme** — küçük harf → URL'leri değiştir → rakamları `"number"` token'ı ile değiştir → noktalama işaretlerini kaldır → boşlukları normalize et
3. **Etiket kodlama** — ham → 0, spam → 1
4. **Bölme** — %80/%20 katmanlı (stratified) train/test ayrımı
5. **Klasik modeller** — TF-IDF (5.000 özellik, İngilizce stop-word'ler) + Naive Bayes / Lojistik Regresyon / Linear SVM / Random Forest
6. **Derin öğrenme** — Keras Tokenizer (10.000 kelime dağarcığı) + pad edilmiş diziler (maksimum uzunluk 100) → Embedding → Bidirectional GRU → özel Attention katmanı → Dense → Sigmoid

## 🏆 Sonuçlar

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes | %97,31 | %100,00 | %79,87 | %88,81 |
| Lojistik Regresyon | %97,67 | %95,56 | %86,58 | %90,85 |
| Linear SVM | %98,48 | %95,21 | %93,29 | %94,24 |
| Random Forest | %98,30 | %100,00 | %87,25 | %93,19 |
| BiGRU | %98,65 | %97,86 | %91,95 | %94,81 |
| BiGRU + Attention | %98,65 | %97,86 | %91,95 | %94,81 |

> ⚠️ **Not:** BiGRU ve BiGRU + Attention satırlarındaki metrikler birebir aynı görünüyor — `model_results.csv` dosyasını kontrol edip attention modelini yeniden değerlendirmen faydalı olur, bu muhtemelen kopyalama kaynaklı bir hata.

**BiGRU + Attention** modeli, precision ve recall arasındaki güçlü dengesi ve attention mekanizmasının sağladığı yorumlanabilirlik avantajı nedeniyle nihai üretim modeli olarak seçilmiştir.

## 🖥️ Uygulamayı Yerelde Çalıştırma

```bash
# 1. Depoyu klonlayın
git clone https://github.com/<kullanici-adiniz>/<repo-adi>.git
cd <repo-adi>

# 2. Sanal ortam oluşturun (önerilir)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Streamlit uygulamasını çalıştırın
streamlit run app.py
```

Ardından terminalde görünen yerel URL'yi (genellikle `http://localhost:8501`) tarayıcınızda açın.

## ⚠️ Sınırlamalar

- Model İngilizce SMS metinleri üzerinde eğitilmiştir — diğer diller veya uzun e-postalardaki performansı değişebilir
- Uygulamadaki "açıklama" paneli, modelin gerçek attention ağırlıkları değil, **kural tabanlı bir anahtar kelime buluşsalıdır (heuristic)** — mekanik bir yorumlama değil, insan tarafından okunabilir bir yardımcı olarak tasarlanmıştır
- Doğrudan kimlik avı (phishing) veya çok dilli tespit için tasarlanmamıştır

## 🛠️ Teknoloji Yığını

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Seaborn / Matplotlib · Streamlit

## 📄 Lisans

Bu proje **MIT Lisansı** altında sunulmuştur — eğitim ve portfolyo amaçlı olarak fork'layabilir, uyarlayabilir ve üzerine geliştirme yapabilirsiniz.

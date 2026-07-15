import pickle
import re
import string
from pathlib import Path

import streamlit as st
import tensorflow as tf
from keras.layers import Layer
from keras.utils import pad_sequences


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "bigru_attention_model.keras"
TOKENIZER_PATH = BASE_DIR / "outputs" / "models" / "tokenizer.pkl"
CONFIG_PATH = BASE_DIR / "outputs" / "models" / "config.pkl"


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Basit "açıklanabilirlik" katmanı: modelin karar verirken tetiklenmiş olabilecek
# yüzeysel ipuçlarını kullanıcıya göstermek için kural tabanlı bir yardımcı.
# Not: Bu, modelin gerçek attention ağırlıklarını göstermez; sadece jüri/kullanıcı
# için insan-okunur bir gerekçe listesi üretir.
# ---------------------------------------------------------------------------
REASON_PATTERNS = {
    "Prize / reward wording": r"\b(win|won|winner|prize|reward|free|gift|voucher)\b",
    "Urgency expressions": r"\b(now|immediately|urgent|hurry|today|expire|limited)\b",
    "Money / financial terms": r"\b(cash|money|\$|price|discount|offer|cost)\b",
    "Call-to-action / links": r"\b(click|call|text|reply|subscribe|url)\b",
    "Numeric codes (phone/amount)": r"\bnumber\b",
}


def get_reasons(cleaned_text, label):
    if label != "Spam":
        return []
    reasons = []
    for reason, pattern in REASON_PATTERNS.items():
        if re.search(pattern, cleaned_text):
            reasons.append(reason)
    return reasons


def get_confidence(probability):
    # Karar sınırından (0.5) ne kadar uzaksa güven o kadar yüksek kabul edilir.
    distance = abs(probability - 0.5) * 2  # 0 -> 1 arası normalize
    if distance >= 0.6:
        return "High"
    if distance >= 0.25:
        return "Medium"
    return "Low"


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False
    )

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, config


def predict_message(message, model, tokenizer, max_len):
    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len)
    probability = float(model.predict(padded, verbose=0)[0][0])
    label = "Spam" if probability >= 0.5 else "Ham"
    return label, probability, cleaned


st.set_page_config(
    page_title="Spam Detection",
    page_icon="🛡️",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(59, 130, 246, 0.22), transparent 30%),
        radial-gradient(circle at 92% 75%, rgba(14, 165, 233, 0.16), transparent 32%),
        linear-gradient(135deg, #eaf4ff 0%, #dbeafe 38%, #eff6ff 100%);
    color: #0f172a;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.3rem;
    padding-bottom: 1.5rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    background: rgba(255, 255, 255, 0.58);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.48);
    border-radius: 30px;
    padding: 38px 42px;
    margin-bottom: 26px;
    box-shadow: 0 20px 50px rgba(37, 99, 235, 0.12);
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    right: -105px;
    top: -115px;
    width: 340px;
    height: 340px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.32), transparent 64%);
}

.hero::after {
    content: "";
    position: absolute;
    left: -95px;
    bottom: -120px;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(37, 99, 235, 0.16), transparent 66%);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #1d4ed8;
    background: rgba(219, 234, 254, 0.80);
    border: 1px solid rgba(96, 165, 250, 0.36);
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 18px;
    position: relative;
    z-index: 1;
}

.hero h1 {
    margin: 0;
    color: #0f172a;
    font-size: 43px;
    font-weight: 900;
    letter-spacing: -0.045em;
    line-height: 1.08;
    position: relative;
    z-index: 1;
}

.hero p {
    margin-top: 16px;
    max-width: 760px;
    color: #334155;
    font-size: 16px;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}

.panel {
    background: rgba(255, 255, 255, 0.56);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.46);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.panel-title {
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 8px;
}

.panel-subtitle {
    color: #475569;
    font-size: 15px;
    margin-bottom: 0;
}

.model-info-card {
    background: rgba(255, 255, 255, 0.54);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.42);
    border-radius: 22px;
    padding: 20px 22px;
    box-shadow: 0 16px 38px rgba(15, 23, 42, 0.07);
}

.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid rgba(100, 116, 139, 0.16);
}

.model-row:last-child {
    border-bottom: none;
}

.model-row span {
    color: #64748b;
    font-size: 14px;
}

.model-row b {
    color: #0f172a;
    font-size: 14px;
    text-align: right;
}

.stTextArea label {
    color: #334155 !important;
    font-weight: 800 !important;
    font-size: 14px !important;
}

.stTextArea textarea {
    background: rgba(255, 255, 255, 0.76) !important;
    color: #0f172a !important;
    border: 1px solid rgba(148, 163, 184, 0.32) !important;
    border-radius: 18px !important;
    min-height: 175px !important;
    font-size: 15px !important;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.stTextArea textarea:focus {
    border: 1px solid rgba(37, 99, 235, 0.62) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
    color: white;
    border: 0;
    border-radius: 16px;
    padding: 0.86rem 1.4rem;
    font-weight: 900;
    font-size: 15px;
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.22);
    transition: 0.2s ease;
}

.stButton > button:hover {
    color: white;
    transform: translateY(-1px);
    background: linear-gradient(135deg, #1d4ed8 0%, #0891b2 100%);
}

.result-wrapper {
    margin-top: 26px;
    background: rgba(255, 255, 255, 0.38);
    border: 1px solid rgba(255, 255, 255, 0.48);
    border-radius: 30px;
    padding: 24px;
    box-shadow: 0 20px 52px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(14px);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    margin-bottom: 18px;
}

.result-header-title {
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
}

.result-header-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-top: 4px;
}

.result-badge {
    background: rgba(37, 99, 235, 0.10);
    color: #1d4ed8;
    border: 1px solid rgba(37, 99, 235, 0.18);
    border-radius: 999px;
    padding: 9px 14px;
    font-size: 13px;
    font-weight: 800;
    white-space: nowrap;
}

.result-grid {
    display: grid;
    grid-template-columns: 1fr 0.9fr;
    gap: 22px;
}

.result-main-card {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    padding: 30px;
    min-height: 250px;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

.result-main-card::before {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    top: -70px;
    right: -70px;
    background: rgba(255, 255, 255, 0.28);
    border-radius: 50%;
}

.result-main-card.spam {
    background: linear-gradient(135deg, #ffe2e2 0%, #fff1f2 100%);
    border: 1px solid rgba(239, 68, 68, 0.20);
}

.result-main-card.ham {
    background: linear-gradient(135deg, #dcfce7 0%, #ecfeff 100%);
    border: 1px solid rgba(34, 197, 94, 0.20);
}

.result-icon {
    font-size: 54px;
    margin-bottom: 16px;
    position: relative;
    z-index: 1;
}

.result-title {
    font-size: 48px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1;
    position: relative;
    z-index: 1;
}

.result-sub {
    margin-top: 18px;
    color: #475569;
    font-size: 16px;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}

.probability-card {
    background: rgba(255, 255, 255, 0.74);
    border-radius: 26px;
    padding: 28px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    min-height: 250px;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
    display: flex;
    flex-direction: column;
}

.probability-label {
    color: #64748b;
    font-size: 15px;
    font-weight: 800;
}

.probability-value {
    margin-top: 20px;
    font-size: 56px;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: -0.04em;
}

.custom-progress {
    width: 100%;
    height: 14px;
    background: #dbeafe;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 28px;
}

.custom-progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
}

.confidence-row {
    margin-top: auto;
    padding-top: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.confidence-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 700;
}

.confidence-pill {
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 800;
}

.confidence-pill.high {
    background: rgba(34, 197, 94, 0.14);
    color: #15803d;
}

.confidence-pill.medium {
    background: rgba(245, 158, 11, 0.16);
    color: #b45309;
}

.confidence-pill.low {
    background: rgba(148, 163, 184, 0.20);
    color: #475569;
}

.reason-list {
    margin-top: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.reason-item {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #334155;
    font-size: 15px;
    background: rgba(37, 99, 235, 0.06);
    border: 1px solid rgba(37, 99, 235, 0.12);
    border-radius: 14px;
    padding: 10px 14px;
}

.preprocess-content {
    color: #334155;
    font-size: 15px;
    line-height: 1.8;
    word-break: break-word;
    background: rgba(148, 163, 184, 0.10);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 8px;
}

.footer-note {
    margin-top: 18px;
    color: #64748b;
    font-size: 12px;
    text-align: center;
}

div[data-testid="stAlert"] {
    border-radius: 16px;
}

@media (max-width: 900px) {
    .result-grid {
        grid-template-columns: 1fr;
    }

    .hero h1 {
        font-size: 34px;
    }

    .result-header {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
<div class="hero-badge">🛡️ Deep Learning Based Text Classification</div>
<h1>Explainable Spam Detection System</h1>
<p>
BiGRU + Attention architecture for spam and ham message classification.
This interface was designed as a modern and presentation-ready demo system.
</p>
</div>
""", unsafe_allow_html=True)


model, tokenizer, config = load_artifacts()
max_len = config["max_len"]

left_col, right_col = st.columns([1.35, 0.65], gap="large")

with left_col:
    st.markdown("""
<div class="panel">
<div class="panel-title">Message Analysis</div>
<div class="panel-subtitle">Enter an English SMS/e-mail message below.</div>
</div>
""", unsafe_allow_html=True)

    user_message = st.text_area(
        "Message Content",
        height=175,
        placeholder="Example: Congratulations! You won a free prize. Click now!"
    )

    analyze_button = st.button("Analyze Message")


with right_col:
    st.markdown("""
<div class="panel">
<div class="panel-title">Model Information</div>
<div class="panel-subtitle">Classification architecture used in the system</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="model-info-card">
<div class="model-row">
<span>Task</span>
<b>Spam / Ham Classification</b>
</div>
<div class="model-row">
<span>Architecture</span>
<b>BiGRU + Attention</b>
</div>
<div class="model-row">
<span>Input</span>
<b>Preprocessed Text</b>
</div>
<div class="model-row">
<span>Output</span>
<b>Prediction + Probability</b>
</div>
</div>
""", unsafe_allow_html=True)


if analyze_button:
    if not user_message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        label, probability, cleaned = predict_message(
            user_message,
            model,
            tokenizer,
            max_len
        )

        result_type = "spam" if label == "Spam" else "ham"
        result_icon = "🚨" if label == "Spam" else "✅"
        result_text = "SPAM" if label == "Spam" else "NOT SPAM"
        result_desc = (
            "This message shows strong characteristics of spam content "
            "(prize offers, urgency, promotional wording, etc.)."
            if label == "Spam"
            else "This message does not show typical spam characteristics "
            "and appears to be legitimate content."
        )

        confidence = get_confidence(probability)
        reasons = get_reasons(cleaned, label)

        reasons_html = "".join(
            f'<div class="reason-item">🔎 {reason}</div>' for reason in reasons
        ) if reasons else '<div class="reason-item">No strong spam-related keyword patterns detected.</div>'

        # NOT: Bu blok artık girintisiz (sütun başından itibaren) yazılıyor.
        # Markdown, 4+ boşluk girintili satırları kod bloğu sayıp HTML render
        # etmediği için önceki sürümde çıktı bozuk görünüyordu. Çözüm buydu.
        result_html = f"""
<div class="result-wrapper">
<div class="result-header">
<div>
<div class="result-header-title">Analysis Result</div>
<div class="result-header-subtitle">
The prediction below is generated by the trained BiGRU + Attention model.
</div>
</div>
<div class="result-badge">Model Prediction</div>
</div>

<div class="result-grid">
<div class="result-main-card {result_type}">
<div class="result-icon">{result_icon}</div>
<div class="result-title">{result_text}</div>
<div class="result-sub">{result_desc}</div>
</div>

<div class="probability-card">
<div class="probability-label">Spam Probability</div>
<div class="probability-value">{probability * 100:.2f}%</div>
<div class="custom-progress">
<div class="custom-progress-fill" style="width:{probability * 100}%;"></div>
</div>
<div class="confidence-row">
<span class="confidence-label">Model Confidence</span>
<span class="confidence-pill {confidence.lower()}">{confidence}</span>
</div>
</div>
</div>
</div>
"""
        st.markdown(result_html, unsafe_allow_html=True)

        # "Neden bu karar verildi?" — jüri/kullanıcı için sade açıklama
        st.markdown(f"""
<div class="panel" style="margin-top: 18px;">
<div class="panel-title" style="font-size: 18px;">Why was this message classified as {result_text.lower()}?</div>
<div class="reason-list">
{reasons_html}
</div>
</div>
""", unsafe_allow_html=True)

        # Teknik detaylar kod bilmeyen kullanıcıdan gizleniyor, isteyen açabilir
        with st.expander("🔍 Technical Details (model input / preprocessing)"):
            st.markdown(f"""
<div class="preprocess-content">{cleaned}</div>
""", unsafe_allow_html=True)
            st.caption("This is the cleaned/preprocessed text that was actually fed into the model.")


st.markdown("""
<div class="footer-note">
Note: This model was trained on English SMS text data. For reliable results, please enter English SMS/e-mail content.
Performance on phishing, domain-specific, or multilingual e-mails may improve with additional training data.
</div>
""", unsafe_allow_html=True)
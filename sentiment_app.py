import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn.functional as F
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# LSTM ARCHITECTURE 
# ─────────────────────────────────────────────

import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.attention = Attention(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        x = self.attention(lstm_out)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PBB Taglish Sentiment Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "all_model_results" not in st.session_state:
    st.session_state.all_model_results = None

st.markdown("""
<style>
.graph-box {
    border: 2px solid #E5E7EB;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
    background-color: white;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
#  GLOBAL CSS  (PBB palette: #1A3FA4 blue,
#               #E8111A red, #F9C800 yellow, white)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&family=Open+Sans:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #F4F6F8;
}

/* ---------- Force white background app-wide ---------- */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
.main, .block-container,
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #FFFFFF !important;
}
/* Bordered containers (st.container(border=True)) should stay white with a light border */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #FFFFFF !important;
}

/* ---------- Safety net: force light rendering even if a visitor's browser/OS
   is set to dark mode (config.toml sets the Streamlit theme itself; this
   backstops the underlying widgets in case a client caches an old theme) --- */
:root, [data-testid="stApp"] {
    color-scheme: light !important;
}
body, [class*="css"], p, span, div, label {
    color: #1e293b;
}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-testid="stSelectbox"] * ,
[data-testid="stExpander"],
[data-testid="stExpander"] summary,
[data-testid="stExpander"] * {
    background-color: #FFFFFF !important;
    color: #1e293b !important;
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #94a3b8 !important;
}

/* ---------- Hide default Streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    padding-top: 0 !important;
}
[data-testid="stMainBlockContainer"],
.main .block-container,
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* ---------- PBB Header Banner ---------- */
.pbb-banner {
    background: linear-gradient(135deg, #1A3FA4 0%, #0d2a7a 50%, #E8111A 100%);
    border-radius: 0;
    padding: 2.2rem 3rem;
    margin: -1rem 0 2rem 0;
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    display: flex;
    align-items: center;
    gap: 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    overflow: hidden;
}
.pbb-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: 8%;
    width: 200px; height: 200px;
    background: rgba(249,200,0,0.15);
    border-radius: 50%;
}
.pbb-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 35%;
    width: 150px; height: 150px;
    background: rgba(249,200,0,0.10);
    border-radius: 50%;
}
.pbb-eye-svg { flex-shrink: 0; }
.pbb-title-block h1 {
    font-family: 'Montserrat', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    color: #FFFFFF;
    margin: 0 0 .25rem 0;
    line-height: 1.1;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
}
.pbb-title-block .subtitle {
    font-family: 'Montserrat', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #F9C800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.pbb-badge {
    background: #F9C800;
    color: #1A3FA4;
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: .7rem;
    letter-spacing: 1px;
    padding: .25rem .7rem;
    border-radius: 20px;
    display: inline-block;
    margin-top: .5rem;
    text-transform: uppercase;
}

/* ---------- Tab Nav (custom) ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #FFFFFF;
    border-radius: 12px;
    padding: 6px;
    box-shadow: 0 2px 12px rgba(26,63,164,.12);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: .82rem;
    letter-spacing: .5px;
    color: #1A3FA4;
    border-radius: 8px;
    padding: .55rem 1.1rem;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1A3FA4, #E8111A) !important;
    color: #FFFFFF !important;
}
/* Force the inner text nodes of the selected tab to white too — Streamlit
   renders the tab label inside a nested <p>/<span>/<div>, and those pick up
   the generic dark body-text color rule above, which otherwise overrides the
   inherited white set on the tab button. */
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ---------- KPI Cards ---------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.kpi-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 4px 18px rgba(26,63,164,.1);
    border-top: 4px solid #1A3FA4;
    transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-4px); }
.kpi-card.red   { border-top-color: #E8111A; }
.kpi-card.yellow{ border-top-color: #F9C800; }
.kpi-card.green { border-top-color: #22c55e; }
.kpi-label {
    font-family: 'Montserrat', sans-serif;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: .4rem;
}
.kpi-value {
    font-family: 'Montserrat', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    color: #1A3FA4;
}
.kpi-card.red   .kpi-value { color: #E8111A; }
.kpi-card.yellow .kpi-value { color: #d4a800; }
.kpi-card.green .kpi-value { color: #16a34a; }
.kpi-sub {
    font-size: .72rem;
    color: #94a3b8;
    margin-top: .2rem;
}

/* ---------- Section Headers ---------- */
.section-header {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #1A3FA4;
    border-left: none;
    padding-left: .75rem;
    margin: 1.5rem 0 1rem 0;
}

/* ---------- Model Info Cards (Home) ---------- */
.model-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.2rem;
    box-shadow: 0 4px 18px rgba(26,63,164,.08);
    border: 2px solid transparent;
    transition: border-color .2s, transform .2s;
    height: 100%;
}
.model-card:hover {
    border-color: #1A3FA4;
    transform: translateY(-3px);
}
.model-tag {
    display: inline-block;
    font-family: 'Montserrat', sans-serif;
    font-size: .65rem;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: .2rem .6rem;
    border-radius: 20px;
    margin-bottom: .5rem;
}
.model-name {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #1e293b;
}
.model-desc {
    font-size: .82rem;
    color: #64748b;
    line-height: 1.55;
    margin-top: .4rem;
}

/* ---------- Sentiment Result ---------- */
.result-box {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 1rem;
    animation: fadeIn .4s ease;
}
.result-box.positive {
    background: linear-gradient(135deg,#dcfce7,#bbf7d0);
    border: 2px solid #22c55e;
}
.result-box.neutral {
    background: linear-gradient(135deg,#dbeafe,#bfdbfe);
    border: 2px solid #1A3FA4;
}
.result-box.negative {
    background: linear-gradient(135deg,#fee2e2,#fecaca);
    border: 2px solid #E8111A;
}
.result-emoji { font-size: 3.5rem; line-height: 1; }
.result-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    margin: .4rem 0 .2rem;
}
.result-box.positive .result-label { color: #16a34a; }
.result-box.neutral  .result-label { color: #1A3FA4; }
.result-box.negative .result-label { color: #E8111A; }
.result-conf {
    font-size: .85rem;
    color: #475569;
}

/* ---------- Confidence Bar ---------- */
.conf-bar-wrap {
    background: #e2e8f0;
    border-radius: 20px;
    height: 10px;
    margin: .4rem 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width .6s ease;
}

/* ---------- Stat Table ---------- */
.stat-pill {
    display: inline-block;
    padding: .15rem .55rem;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 700;
}
.reject { background:#fee2e2; color:#E8111A; }
.fail   { background:#dcfce7; color:#16a34a; }

/* ---------- Objective List ---------- */
.obj-item {
    display: flex;
    align-items: flex-start;
    gap: .75rem;
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: .75rem;
    box-shadow: 0 2px 10px rgba(26,63,164,.07);
}
.obj-num {
    background: linear-gradient(135deg,#1A3FA4,#E8111A);
    color: white;
    font-family: 'Montserrat', sans-serif;
    font-weight: 900;
    font-size: .9rem;
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.obj-text { font-size: .88rem; color: #334155; line-height: 1.5; }

/* ---------- Animations ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);     }
}

/* ---------- Info Box ---------- */
.info-box {
    background: linear-gradient(135deg, #EFF6FF, #FEF9C3);
    border-left: 5px solid #F9C800;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    font-size: .85rem;
    color: #334155;
    line-height: 1.6;
}

/* ---------- Buttons (Analyzer panel + examples) ---------- */
.stButton > button {
    border-radius: 8px;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: .85rem;
}

/* ---------- Findings list (Key Findings, plain text style) ---------- */
.findings-block {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 18px rgba(26,63,164,.08);
}
.findings-item {
    padding: 1rem 0;
    border-bottom: 1px solid #E5E7EB;
}
.findings-item:last-child { border-bottom: none; }
.findings-tag {
    display: inline-block;
    font-family: 'Montserrat', sans-serif;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .8px;
    text-transform: uppercase;
    padding: .2rem .6rem;
    border-radius: 20px;
    margin-bottom: .5rem;
}
.findings-text {
    font-size: .88rem;
    color: #334155;
    line-height: 1.65;
    margin: 0;
}

/* ---------- Footer ---------- */
.pbb-footer {
    text-align: center;
    padding: 1.5rem;
    font-size: .75rem;
    color: #94a3b8;
    border-top: 2px solid #e2e8f0;
    margin-top: 3rem;
    font-family: 'Montserrat', sans-serif;
    letter-spacing: .5px;
}
</style>
""", unsafe_allow_html=True)

# ===== MODELS
@st.cache_resource
def load_all_models():

    device = torch.device("cpu")

    # LSTM
    with open("models/lstm/lstm_tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)
    
    with open("models/lstm/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    vocab_size = len(lstm_tokenizer.word_index) + 1
    num_classes = len(label_encoder.classes_)

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=200,
        hidden_dim=128,
        num_classes=num_classes
    )

    lstm_model.load_state_dict(
        torch.load("models/lstm/lstm_sentiment_model.pt", map_location=device)
    )

    lstm_model.to(device)
    lstm_model.eval()

    # mBERT
    mbert_tok = AutoTokenizer.from_pretrained("models/mbert")
    mbert_model = AutoModelForSequenceClassification.from_pretrained("models/mbert").to(device)
    mbert_model.eval()

    # XLM-R
    xlm_tok = AutoTokenizer.from_pretrained("models/xlmr")
    xlm_model = AutoModelForSequenceClassification.from_pretrained("models/xlmr").to(device)
    xlm_model.eval()

    # RoBERTa
    tl_tok = AutoTokenizer.from_pretrained("models/roberta")
    tl_model = AutoModelForSequenceClassification.from_pretrained("models/roberta").to(device)
    tl_model.eval()

    return (
        lstm_model, lstm_tokenizer, label_encoder,
        mbert_tok, mbert_model,
        xlm_tok, xlm_model,
        tl_tok, tl_model
    )

# ─────────────────────────────────────────────
#  HEADER BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="pbb-banner">
  <!-- PBB Eye SVG -->
  <svg class="pbb-eye-svg" width="100" height="70" viewBox="0 0 200 140" xmlns="http://www.w3.org/2000/svg">
    <!-- outer eye shape -->
    <ellipse cx="100" cy="70" rx="95" ry="58" fill="#E8111A"/>
    <ellipse cx="100" cy="70" rx="80" ry="45" fill="#1A3FA4"/>
    <!-- iris -->
    <circle cx="100" cy="70" r="35" fill="#F9C800"/>
    <!-- pupil / sun symbol -->
    <circle cx="100" cy="70" r="20" fill="#1A3FA4"/>
    <!-- sun rays -->
    <line x1="100" y1="42" x2="100" y2="35" stroke="#F9C800" stroke-width="3" stroke-linecap="round"/>
    <line x1="100" y1="98" x2="100" y2="105" stroke="#F9C800" stroke-width="3" stroke-linecap="round"/>
    <line x1="72" y1="70" x2="65"  y2="70" stroke="#F9C800" stroke-width="3" stroke-linecap="round"/>
    <line x1="128" y1="70" x2="135" y2="70" stroke="#F9C800" stroke-width="3" stroke-linecap="round"/>
    <line x1="80"  y1="50" x2="75"  y2="45" stroke="#F9C800" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="120" y1="90" x2="125" y2="95" stroke="#F9C800" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="120" y1="50" x2="125" y2="45" stroke="#F9C800" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="80"  y1="90" x2="75"  y2="95" stroke="#F9C800" stroke-width="2.5" stroke-linecap="round"/>
    <!-- highlight -->
    <circle cx="110" cy="60" r="6" fill="rgba(255,255,255,0.35)"/>
  </svg>

  <div class="pbb-title-block">
    <h1>PBB Taglish Sentiment Analyzer</h1>
    <div class="subtitle">Analyzing Taglish Fan Discussions from Reddit</div>
    <div class="pbb-badge"> Sentiment Analysis · Natural Language Processing · Code-switched Text</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────
MODELS = ["LSTM", "mBERT", "RoBERTa", "XLM-RoBERTa"]
MODEL_COLORS = {
    "LSTM":        "#F9C800",
    "mBERT":       "#1A3FA4",
    "XLM-RoBERTa": "#E8111A",
    "RoBERTa":     "#8B5CF6",
}

metrics_data = {
    "Model": MODELS,
    "Accuracy":  [0.84, 0.85, 0.86, 0.89],
    "Precision": [0.84, 0.88, 0.86, 0.89],
    "Recall":    [0.84, 0.85, 0.86, 0.89],
    "F1-Score":  [0.85, 0.86, 0.86, 0.89],
}
df_metrics = pd.DataFrame(metrics_data)

# Per-class accuracy 
per_class = {
    "Model":     MODELS,
    "Positive":  [0.91, 0.99, 0.96, 0.95],
    "Neutral":   [0.75, 0.73, 0.79, 0.84],
    "Negative":  [0.86, 0.92, 0.84, 0.88],
}
df_class = pd.DataFrame(per_class)

def lstm_predict(text):

    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=150)

    x = torch.tensor(padded).long()

    with torch.no_grad():
        outputs = lstm_model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    pred = np.argmax(probs)
    label = label_encoder.inverse_transform([pred])[0]

    return label, probs

def transformer_predict(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

    pred = np.argmax(probs)
    label = label_encoder.inverse_transform([pred])[0]

    return label, probs
(
    lstm_model, lstm_tokenizer, label_encoder,
    mbert_tok, mbert_model,
    xlm_tok, xlm_model,
    tl_tok, tl_model
) = load_all_models()

def predict(text, model_name):

    if model_name == "LSTM":
        return lstm_predict(text)

    elif model_name == "mBERT":
        return transformer_predict(text, mbert_tok, mbert_model)

    elif model_name == "RoBERTa":
        return transformer_predict(text, tl_tok, tl_model)

    elif model_name == "XLM-RoBERTa":
        return transformer_predict(text, xlm_tok, xlm_model)
# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Model Evaluation",
    "Sentiment Analyzer"
])

# ══════════════════════════════════════════════
#  TAB 1 — INTRODUCTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">About the Study</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <strong>Research Overview</strong><br>
    This website explores <strong>Taglish (Tagalog–English) discussions</strong> from Reddit 
    related to <em>Pinoy Big Brother: Celebrity Collab Edition Season 1</em>. Posts and comments from 
    the <strong>r/pinoybigbrother</strong> community were collected and analyzed using 
    sentiment classification to determine whether discussions express 
    <strong>Positive</strong>, <strong>Neutral</strong>, or <strong>Negative</strong> reactions.
    The study compares four different models —
    <strong>LSTM, mBERT, RoBERTa, and XLM-RoBERTa</strong> — to evaluate how different 
    architectures interpret mixed Tagalog–English social media conversations.
    </div>""", unsafe_allow_html=True)


    # Dataset Info
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.9rem; color:#475569; line-height:1.6;">
    <strong>Big 4 Duos in the Dataset</strong><br>
    Reddit discussions collected for this study mainly revolve around the 
    <strong>Big 4 duos from Pinoy Big Brother: Celebrity Collab Season 1</strong>. 
    Posts and comments mentioning these pairs were grouped to observe sentiment 
    patterns among fans in the <em>r/pinoybigbrother</em> community.
    </p>""", unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.image("images/azver.jpg", use_container_width=True)
        st.markdown("""
        <div style="font-family:Montserrat;font-weight:800;font-size:1.05rem;color:#1e293b;">
        AzVer
        </div>
        <div style="font-size:.82rem;color:#64748b;">
        Azriel 'AZ' Martinez & River Joseph
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.image("images/breka.jpg", use_container_width=True)
        st.markdown("""<div style="font-family:Montserrat;font-weight:800;font-size:1.05rem;color:#1e293b;">
        BreKa
        </div>
        <div style="font-size:.82rem;color:#64748b;">
        Brent Manalo & Mika Salamanca
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.image("images/chares.jpg", use_container_width=True)
        st.markdown("""<div style="font-family:Montserrat;font-weight:800;font-size:1.05rem;color:#1e293b;">
        CharEs
        </div>
        <div style="font-size:.82rem;color:#64748b;">
        Charlie Flemming & Esnyr Ranollo
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        st.image("images/rawi.jpg", use_container_width=True)
        st.markdown("""<div style="font-family:Montserrat;font-weight:800;font-size:1.05rem;color:#1e293b;">
        RaWi
        </div>
        <div style="font-size:.82rem;color:#64748b;">
        Ralph de Leon & Will Ashley de Leon
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    These duo names were used as keywords to group Reddit discussions in the dataset. 
    By analyzing posts and comments mentioning these pairs, the system evaluates how 
    fans express their opinions and reactions toward contestants using Taglish language.
    """, unsafe_allow_html=True)

    # Model Cards
    st.markdown('<div class="section-header">Models Compared</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)

    model_info = [
        ("LSTM", "#F9C800", "#1A3FA4", "Baseline Model",
         "Long Short-Term Memory (LSTM) is a recurrent neural network architecture capable of learning long-range dependencies in sequential data. It serves as the baseline traditional deep learning model in this study."),
        ("mBERT", "#1A3FA4", "#FFFFFF", "Multilingual Transformer",
         "Multilingual BERT (mBERT) is pre-trained on 104 languages, making it well-suited for Taglish text. It uses bidirectional attention to capture rich contextual representations from both Tagalog and English tokens."),
        ("XLM-RoBERTa", "#E8111A", "#FFFFFF", "Cross-lingual Transformer",
         "XLM-RoBERTa is a cross-lingual transformer pre-trained on a massive multilingual corpus with a robust training objective, offering strong performance on low-resource and code-switched language tasks."),
        ("RoBERTa", "#8B5CF6", "#FFFFFF", "Optimized Transformer",
         "RoBERTa is an optimized variant of BERT with improved pre-training strategies including dynamic masking and larger batch sizes. Its strong English understanding may benefit the English portions of Taglish."),
    ]
    for col, (name, bg, fg, tag, desc) in zip([mc1, mc2, mc3, mc4], model_info):
        with col:
            st.markdown(f"""
            <div class="model-card">
                <span class="model-tag" style="background:{bg};color:{fg};">{tag}</span>
                <div class="model-name">{name}</div>
                <div class="model-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Key Findings — plain, clean text format (no cards)
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)

    findings = [
        ("RQ1 · Model Comparison", "#1A3FA4", "#FFFFFF",
         "Pre-trained transformers outperformed the trained-from-scratch LSTM baseline. "
         "XLM-RoBERTa led with 0.89 accuracy / weighted F1, followed by RoBERTa (0.86), "
         "mBERT (0.85), and LSTM (0.84). LSTM and mBERT showed a bias toward neutral labels "
         "on ambiguous text, while RoBERTa and XLM-RoBERTa captured contextual nuance more reliably."),
        ("RQ2 · Code-Switching", "#F9C800", "#1A3FA4",
         "Every model performed better on a pure-English baseline than on the 98.63% "
         "code-switched PBB dataset. LSTM had the largest accuracy drop on Taglish (-0.08), "
         "while XLM-RoBERTa was the most resilient (-0.02) — evidence that multilingual "
         "pre-training helps with code-switched text."),
        ("RQ3 · Sentiment Spread", "#E8111A", "#FFFFFF",
         "mBERT produced the most neutral-heavy predictions (45.4% neutral), often defaulting "
         "there on ambiguous cues. XLM-RoBERTa produced the most polarized, confident "
         "predictions, surfacing more of the fanbase's real positive and negative reactions "
         "across the Big 4 duos."),
        ("RQ4 · Voting Alignment", "#8B5CF6", "#FFFFFF",
         "All four models predicted the same ranking — RaWi 1st, BreKa 2nd, CharEs 3rd, AzVer 4th — "
         "correctly matching the official 3rd- and 4th-place duos (CharEs 22.91%, AzVer 8.77%), with "
         "predicted shares landing close to the actual vote totals. The top two placements were "
         "reversed: BreKa actually won with 33.03% of the vote against RaWi's 25.88%, while every model "
         "favored RaWi instead. This suggests Reddit sentiment tracks general audience support well, "
         "but can't fully capture the show's unlimited financial-voting mechanic, which likely tipped "
         "the final result in BreKa's favor."),
    ]

    items_html = "".join([
        f'<div class="findings-item">'
        f'<span class="findings-tag" style="background:{bg};color:{fg};">{tag}</span>'
        f'<p class="findings-text">{text}</p>'
        f'</div>'
        for tag, bg, fg, text in findings
    ])
    st.markdown(f'<div class="findings-block">{items_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pbb-footer">
        PBB Taglish Sentiment Analyzer
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — MODEL EVALUATION
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Overall Performance Metrics</div>',
                unsafe_allow_html=True)

    # ── KPI row: best model per metric ──────────────
    best_acc   = df_metrics.loc[df_metrics["Accuracy"].idxmax()]
    best_f1    = df_metrics.loc[df_metrics["F1-Score"].idxmax()]
    best_prec  = df_metrics.loc[df_metrics["Precision"].idxmax()]
    best_rec   = df_metrics.loc[df_metrics["Recall"].idxmax()]

    k1, k2, k3, k4 = st.columns(4)
    for col, label, model_row, metric, card_cls in [
        (k1, "Best Accuracy",   best_acc,  "Accuracy",  ""),
        (k2, "Best F1-Score",   best_f1,   "F1-Score",  "green"),
        (k3, "Best Precision",  best_prec, "Precision", ""),
        (k4, "Best Recall",     best_rec,  "Recall",    "red"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {card_cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="{'color:#16a34a;' if card_cls=='green' else ''}">
                    {model_row[metric]:.2f}
                </div>
                <div class="kpi-sub">{model_row['Model']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: metrics table + bar chart side by side, each in its own card ──
    left, right = st.columns([2, 3])
    CARD_HEIGHT = 360

    with left:
        with st.container(border=True, height=CARD_HEIGHT):
            st.markdown("**Metrics Summary**")
            th_style = "padding:10px 8px;font-family:'Montserrat',sans-serif;font-size:.68rem;letter-spacing:.6px;text-transform:uppercase;color:#64748b;"
            header_html = (
                '<tr style="border-bottom:2px solid #E5E7EB;">'
                f'<th style="{th_style}text-align:left;">Model</th>'
                f'<th style="{th_style}text-align:center;">Accuracy</th>'
                f'<th style="{th_style}text-align:center;">Precision</th>'
                f'<th style="{th_style}text-align:center;">Recall</th>'
                f'<th style="{th_style}text-align:center;">F1</th>'
                '</tr>'
            )

            row_htmls = []
            for _, row in df_metrics.iterrows():
                dot = MODEL_COLORS[row["Model"]]
                row_htmls.append(
                    '<tr style="border-bottom:1px solid #F1F5F9;">'
                    '<td style="padding:10px 8px;white-space:nowrap;">'
                    f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{dot};margin-right:8px;"></span>'
                    f'<strong style="color:#1e293b;">{row["Model"]}</strong></td>'
                    f'<td style="padding:10px 8px;text-align:center;color:#334155;">{row["Accuracy"]:.0%}</td>'
                    f'<td style="padding:10px 8px;text-align:center;color:#334155;">{row["Precision"]:.0%}</td>'
                    f'<td style="padding:10px 8px;text-align:center;color:#334155;">{row["Recall"]:.0%}</td>'
                    f'<td style="padding:10px 8px;text-align:center;color:#334155;">{row["F1-Score"]:.0%}</td>'
                    '</tr>'
                )

            table_html = (
                '<div style="height:280px;display:flex;flex-direction:column;justify-content:center;">'
                '<table style="width:100%;border-collapse:collapse;font-family:\'Open Sans\',sans-serif;font-size:.85rem;">'
                f'<thead>{header_html}</thead>'
                f'<tbody>{"".join(row_htmls)}</tbody>'
                '</table></div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

    with right:
        with st.container(border=True, height=CARD_HEIGHT):
            st.markdown("**Per-Class Accuracy Breakdown**")
            fig_line = go.Figure()
            for cls, color in [("Positive", "#1A3FA4"), ("Neutral", "#F9C800"), ("Negative", "#E8111A")]:
                fig_line.add_trace(go.Scatter(
                    x=df_class["Model"], y=df_class[cls],
                    mode="lines+markers+text",
                    name=cls,
                    line=dict(color=color, width=3),
                    marker=dict(size=9, symbol="circle"),
                    text=[f"{v:.0%}" for v in df_class[cls]],
                    textposition="top center",
                    textfont=dict(size=10, family="Montserrat"),
                ))
            fig_line.update_layout(
                showlegend=False,
                yaxis=dict(range=[0.6, 1.05], tickformat=".0%", gridcolor="#e2e8f0"),
                plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Montserrat", size=11),
                margin=dict(t=10, b=20, l=20, r=20),
                height=280,
            )
            st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Read the detailed model-by-model analysis"):
        st.markdown("""
        Looking at all four models, a clear pattern emerges: they are highly effective at identifying **:green[Positive sentiments]**, perform well with **:red[Negative sentiments]**, but face the greatest difficulty with **:gray[Neutral comments]**. Positive posts are generally easier to classify because fans often use direct and clear language to show support. Neutral posts, on the other hand, are much more complex, frequently containing mixed opinions, questions, or general show updates. Combined with the use of code-switched Taglish, sarcasm, and slang, these factors make it difficult for the models to determine the definitive sentiment of a comment.

        Here is a breakdown of how each model processed these linguistic challenges:

        **LSTM - The Baseline Classifier**
        * It was trained entirely from scratch, meaning it relies heavily on the sequential word order it learned directly from the dataset.
        * Lacking the advantage of massive pre-trained language data, typical Reddit slang and irregular spellings often disrupt its predictions.
        * As a result, it frequently misclassifies subtle positivity or negativity as a neutral tone, keeping its neutral precision down at 75%.
            
        **mBERT - The Conservative Classifier**
        * mBERT exhibits a highly cautious classification style. When it successfully identifies a positive comment, it is almost always correct, achieving a 99% precision rate.
        * However, when it encounters sarcastic or confusing Taglish, it lacks the deep contextual understanding required to interpret it and defaults to labeling it neutral.
        * This tendency to assign ambiguous text to the neutral category drops its neutral precision significantly to 73%.
  
        **RoBERTa - The Intermediate Step**
        * RoBERTa handles complex text better by breaking down unfamiliar Taglish slang into smaller, recognizable subword units.
        * This capability allows it to confidently identify positive and negative emotions rather than defaulting to neutral.
        * Nevertheless, it still struggles with implied criticism, occasionally confusing neutral and negative labels when a user's disapproval is indirect rather than clearly stated.

        **XLM-RoBERTa - The Top Performer**
        * This model delivers the most balanced and consistent predictions across all sentiment categories.
        * Because it was extensively pre-trained on multiple languages, it effectively interprets the context of mixed Tagalog and English sentences without requiring translation.
        * By accurately classifying the complex, polarized sentences that confused the other models, it raises the neutral precision to 84%.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # RQ2 — CODE-SWITCHING IMPACT (from the defense results)
    # ══════════════════════════════════════════════
    st.markdown('<div class="section-header">Code-Switching Impact — RQ2</div>', unsafe_allow_html=True)

    codeswitch_data = {
        "Model": MODELS,
        "Code-switched": [0.84, 0.85, 0.86, 0.89],
        "Pure English":  [0.92, 0.90, 0.91, 0.91],
    }
    df_cs = pd.DataFrame(codeswitch_data)

    with st.container(border=True):
        st.markdown("**Accuracy: Code-switched PBB Data vs. Pure English Data**")
        st.markdown(
            '<div style="font-size:.78rem;color:#94a3b8;margin:-4px 0 10px 0;">'
            'Pure-English baseline: a 37k-entry Kaggle dataset on the 2019 Indian General Elections. '
            'PBB dataset: 8,048 Reddit posts/comments (Jun 7–Jul 5, 2025), 98.63% code-switched Taglish.</div>',
            unsafe_allow_html=True,
        )
        fig_cs = go.Figure()
        fig_cs.add_trace(go.Bar(name="Code-switched (PBB)", x=df_cs["Model"], y=df_cs["Code-switched"],
                                 marker_color="#F9C800", text=[f"{v:.2f}" for v in df_cs["Code-switched"]],
                                 textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color="#1A3FA4", size=13, family="Montserrat")))
        fig_cs.add_trace(go.Bar(name="Pure English", x=df_cs["Model"], y=df_cs["Pure English"],
                                 marker_color="#1A3FA4", text=[f"{v:.2f}" for v in df_cs["Pure English"]],
                                 textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color="#FFFFFF", size=13, family="Montserrat")))
        fig_cs.update_layout(
            barmode="group",
            yaxis=dict(range=[0,1], tickformat=".0%", gridcolor="#e2e8f0"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Montserrat", size=12),
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=10, b=20, l=20, r=20),
            height=320,
        )
        st.plotly_chart(fig_cs, use_container_width=True)

        deltas = [round(b - a, 2) for a, b in zip(df_cs["Code-switched"], df_cs["Pure English"])]
        dcols = st.columns(4)
        for c, m, d in zip(dcols, df_cs["Model"], deltas):
            with c:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<span style="background:#dcfce7;color:#16a34a;font-weight:800;font-family:\'Montserrat\',sans-serif;'
                    f'font-size:.8rem;padding:.25rem .6rem;border-radius:20px;">+{d:.2f}</span>'
                    f'<div style="font-size:.72rem;color:#64748b;margin-top:4px;">{m}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div style="background:#F8FAFC;border:1px solid #E5E7EB;border-radius:10px;'
            'padding:.9rem 1.1rem;margin-top:1rem;font-size:.82rem;color:#475569;line-height:1.6;">'
            'All four models perform better on English-only data, confirming that Taglish code-switching '
            'increases classification difficulty. LSTM shows the largest accuracy gap (+0.08), while '
            'XLM-RoBERTa is the most resilient to code-switching (+0.02).</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # RQ3 — PREDICTED SENTIMENT DISTRIBUTION
    # ══════════════════════════════════════════════
    st.markdown('<div class="section-header">Predicted Sentiment Distribution — RQ3</div>', unsafe_allow_html=True)

    dist_data = {
        "Model":    ["LSTM", "mBERT", "RoBERTa", "XLM-RoBERTa"],
        "Positive": [31.4, 28.7, 31.9, 33.0],
        "Neutral":  [40.0, 45.4, 37.9, 36.5],
        "Negative": [28.6, 25.9, 30.3, 30.6],
    }
    df_dist = pd.DataFrame(dist_data)

    with st.container(border=True):
        st.markdown("**Predicted Sentiment Class Distribution per Model**")
        st.markdown(
            '<div style="font-size:.78rem;color:#94a3b8;margin:-4px 0 10px 0;">'
            'Based on confusion matrices across the full PBB dataset.</div>',
            unsafe_allow_html=True,
        )
        fig_dist = go.Figure()
        for cls, color in [("Positive", "#1A3FA4"), ("Neutral", "#F9C800"), ("Negative", "#E8111A")]:
            fig_dist.add_trace(go.Bar(
                name=cls, y=df_dist["Model"], x=df_dist[cls], orientation="h",
                marker_color=color, text=[f"{v:.1f}%" for v in df_dist[cls]],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=12, family="Montserrat"),
            ))
        fig_dist.update_layout(
            barmode="stack",
            xaxis=dict(range=[0,100], ticksuffix="%", gridcolor="#e2e8f0"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Montserrat", size=12),
            showlegend=False,
            margin=dict(t=10, b=20, l=20, r=20),
            height=260,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        kk1, kk2 = st.columns(2, gap="medium")
        with kk1:
            st.markdown(
                '<div style="border-left:5px solid #1A3FA4;border-radius:12px;background:#EFF6FF;'
                'padding:1rem 1.2rem;font-size:.85rem;color:#334155;line-height:1.6;'
                'box-shadow:0 2px 10px rgba(26,63,164,.07);margin-bottom:.5rem;">'
                '<strong>Most neutral-heavy: mBERT</strong><br>45.4% of its predictions land in Neutral — it tends to '
                'default there on ambiguous Taglish cues, masking real polarity.</div>',
                unsafe_allow_html=True,
            )
        with kk2:
            st.markdown(
                '<div style="border-left:5px solid #E8111A;border-radius:12px;background:#FEF2F2;'
                'padding:1rem 1.2rem;font-size:.85rem;color:#334155;line-height:1.6;'
                'box-shadow:0 2px 10px rgba(232,17,26,.07);margin-bottom:.5rem;">'
                '<strong>Most polarized: XLM-RoBERTa</strong><br>Reduces false neutrals and surfaces more of the '
                'show\'s real positive/negative reactions.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RQ3 continued: per-duo breakdown ─────────────────────────────────────
    duo_data = {
        "LSTM":        {"AzVer": (45.1, 7.4, 47.5),  "BreKa": (39.2, 35.4, 25.4), "CharEs": (36.4, 33.7, 29.9), "RaWi": (40.3, 46.1, 13.6)},
        "mBERT":       {"AzVer": (51.2, 1.3, 47.5),  "BreKa": (40.7, 37.9, 21.4), "CharEs": (33.7, 34.4, 31.8), "RaWi": (36.8, 58.2, 5.0)},
        "RoBERTa":     {"AzVer": (10.9, 38.7, 50.4), "BreKa": (40.5, 36.9, 22.6), "CharEs": (34.2, 30.9, 34.9), "RaWi": (40.1, 45.1, 14.8)},
        "XLM-RoBERTa": {"AzVer": (16.3, 34.4, 49.3), "BreKa": (40.7, 36.9, 22.4), "CharEs": (33.7, 34.4, 31.8), "RaWi": (39.9, 39.9, 20.3)},
    }
    duo_names = {"AzVer": "AzVer (AZ Martinez & River Joseph)",
                 "BreKa": "BreKa (Brent Manalo & Mika Salamanca)",
                 "CharEs": "CharEs (Charlie Flemming & Esnyr Ranollo)",
                 "RaWi": "RaWi (Ralph & Will de Leon)"}

    with st.container(border=True):
        st.markdown("**Sentiment Distribution by Duo**")
        duo_choice = st.selectbox("Select a duo", list(duo_names.keys()),
                                   format_func=lambda k: duo_names[k], key="duo_pick",
                                   label_visibility="collapsed")
        pos = [duo_data[m][duo_choice][0] for m in MODELS]
        neu = [duo_data[m][duo_choice][1] for m in MODELS]
        neg = [duo_data[m][duo_choice][2] for m in MODELS]

        fig_duo = go.Figure()
        fig_duo.add_trace(go.Bar(name="Positive", y=MODELS, x=pos, orientation="h",
                                  marker_color="#1A3FA4", text=[f"{v:.1f}%" for v in pos],
                                  textposition="inside", insidetextanchor="middle"))
        fig_duo.add_trace(go.Bar(name="Neutral", y=MODELS, x=neu, orientation="h",
                                  marker_color="#F9C800", text=[f"{v:.1f}%" for v in neu],
                                  textposition="inside", insidetextanchor="middle"))
        fig_duo.add_trace(go.Bar(name="Negative", y=MODELS, x=neg, orientation="h",
                                  marker_color="#E8111A", text=[f"{v:.1f}%" for v in neg],
                                  textposition="inside", insidetextanchor="middle"))
        fig_duo.update_layout(
            barmode="stack",
            xaxis=dict(range=[0,100], ticksuffix="%", gridcolor="#e2e8f0"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Montserrat", size=12),
            showlegend=False,
            margin=dict(t=10, b=20, l=20, r=20),
            height=260,
        )
        st.plotly_chart(fig_duo, use_container_width=True)
        st.markdown(
            '<div style="background:#F8FAFC;border:1px solid #E5E7EB;border-radius:10px;'
            'padding:.8rem 1rem;margin-top:.8rem;font-size:.75rem;color:#64748b;line-height:1.6;">'
            'Highlights across duos: BreKa has the highest positive share (40.7%) under mBERT and XLM-RoBERTa; '
            'RaWi has the highest neutral share (58.2%) under mBERT; AzVer has the highest negative share (50.4%) under RoBERTa. '
            'A few very small segments (~1–5%) were not individually labeled in the original charts and are shown here '
            'as the remainder needed to reach 100%.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # RQ4 — VOTING PREDICTION ALIGNMENT
    # ══════════════════════════════════════════════
    st.markdown('<div class="section-header">Voting Prediction Alignment — RQ4</div>', unsafe_allow_html=True)

    voting_data = {
        "Duo":            ["BreKa",  "RaWi",   "CharEs", "AzVer"],
        "Official Rank":  [1, 2, 3, 4],
        "Official %":     [33.03, 25.88, 22.91, 8.77],
        "LSTM":           [27.74, 36.76, 24.69, 10.81],
        "mBERT":          [27.75, 43.44, 20.91, 7.89],
        "RoBERTa":        [29.97, 35.81, 22.35, 11.87],
        "XLM-RoBERTa":    [30.71, 31.79, 23.64, 13.86],
    }
    df_vote = pd.DataFrame(voting_data)
    predicted_rank_map = {"RaWi": 1, "BreKa": 2, "CharEs": 3, "AzVer": 4}

    with st.container(border=True):
        st.markdown("**Predicted Vote Share vs. Official Big Night Results**")
        st.markdown(
            '<div style="font-size:.78rem;color:#94a3b8;margin:-4px 0 10px 0;">'
            'Aggregated Reddit sentiment per duo, mapped to a predicted vote share and ranking, '
            'compared against the official Big Night results.</div>',
            unsafe_allow_html=True,
        )

        fig_vote = go.Figure()
        fig_vote.add_trace(go.Bar(
            name="Official Result", x=df_vote["Duo"], y=df_vote["Official %"],
            marker_color="#1e293b",
            text=[f"{v:.2f}%" for v in df_vote["Official %"]],
            textposition="outside", textfont=dict(size=11, family="Montserrat"),
        ))
        for model in MODELS:
            fig_vote.add_trace(go.Bar(
                name=model, x=df_vote["Duo"], y=df_vote[model],
                marker_color=MODEL_COLORS[model],
                text=[f"{v:.2f}%" for v in df_vote[model]],
                textposition="outside", textfont=dict(size=11, family="Montserrat"),
            ))
        fig_vote.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 50], ticksuffix="%", gridcolor="#e2e8f0", title="Vote Share"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Montserrat", size=12),
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=10, b=20, l=20, r=20),
            height=380,
        )
        st.plotly_chart(fig_vote, use_container_width=True)

        # Ranking comparison table
        th_style = "padding:10px 8px;font-family:'Montserrat',sans-serif;font-size:.68rem;letter-spacing:.6px;text-transform:uppercase;color:#64748b;"
        header_html = (
            '<tr style="border-bottom:2px solid #E5E7EB;">'
            f'<th style="{th_style}text-align:left;">Duo</th>'
            f'<th style="{th_style}text-align:center;">Official Rank</th>'
            f'<th style="{th_style}text-align:center;">LSTM</th>'
            f'<th style="{th_style}text-align:center;">mBERT</th>'
            f'<th style="{th_style}text-align:center;">RoBERTa</th>'
            f'<th style="{th_style}text-align:center;">XLM-RoBERTa</th>'
            '</tr>'
        )
        row_htmls = []
        for _, row in df_vote.iterrows():
            duo = row["Duo"]
            pred_rank = predicted_rank_map[duo]
            match = pred_rank == row["Official Rank"]
            pill_bg, pill_fg = ("#dcfce7", "#16a34a") if match else ("#fee2e2", "#E8111A")
            cells = "".join([
                f'<td style="padding:10px 8px;text-align:center;">'
                f'<span style="background:{pill_bg};color:{pill_fg};font-weight:800;'
                f'font-size:.75rem;padding:.15rem .55rem;border-radius:20px;">#{pred_rank} · {row[m]:.2f}%</span></td>'
                for m in MODELS
            ])
            row_htmls.append(
                '<tr style="border-bottom:1px solid #F1F5F9;">'
                f'<td style="padding:10px 8px;"><strong style="color:#1e293b;">{duo}</strong></td>'
                f'<td style="padding:10px 8px;text-align:center;color:#334155;">#{row["Official Rank"]} ({row["Official %"]:.2f}%)</td>'
                f'{cells}'
                '</tr>'
            )
        table_html = (
            '<div style="overflow-x:auto;margin-top:1rem;">'
            '<table style="width:100%;border-collapse:collapse;font-family:\'Open Sans\',sans-serif;font-size:.85rem;">'
            f'<thead>{header_html}</thead>'
            f'<tbody>{"".join(row_htmls)}</tbody>'
            '</table></div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.72rem;color:#94a3b8;margin-top:.5rem;">'
            'Green pill = predicted rank matches the official rank for that duo; red pill = mismatch. '
            'All models agree on the ranking order (RaWi 1st, BreKa 2nd, CharEs 3rd, AzVer 4th).</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="background:#F8FAFC;border:1px solid #E5E7EB;border-radius:10px;'
            'padding:.9rem 1.1rem;margin-top:1rem;font-size:.82rem;color:#475569;line-height:1.6;">'
            'All four models correctly identified CharEs (3rd) and AzVer (4th), with predicted vote shares '
            'landing close to the actual results — mBERT\'s 7.89% for AzVer was the closest single estimate to the '
            'actual 8.77%. Every model, however, predicted RaWi to win over BreKa, reversing the actual 1st- and '
            '2nd-place finish (BreKa 33.03% vs. RaWi 25.88%). The predicted top-two vote shares were still close in '
            'magnitude to the real totals, just assigned to the wrong duo — a likely sign that Reddit sentiment '
            'tracks general audience support well, but can\'t capture the show\'s unlimited financial-voting '
            'mechanic via the Maya app, which appears to have driven BreKa\'s win.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="pbb-footer"> </div>
    """, unsafe_allow_html=True)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
# ══════════════════════════════════════════════
#  TAB 3 — INTERACTIVE SENTIMENT ANALYZER
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Try the Taglish Sentiment Analyzer</div>',
                unsafe_allow_html=True)

    # Banner tip
    st.markdown("""
    <div class="info-box">
        <strong>Instructions:</strong> Select one of the example templates below, or type your own
        Taglish sentence or phrase related to PBB. Select which model to use, then click
        <strong>Analyze Sentiment</strong> to see how it predicts your sentiment, complete with a
        confidence score.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Example prompts ──────
    st.markdown("**Example PBB-related Taglish inputs:**")
    examples = [
        "Sobrang galing ni [Housemate]! Deserve niya ang immunity for the week!",
        "May bagong task na ibinigay sa mga housemates ngayong episode.",
        "Ang disappointing ng naging strategy ni [Housemate], lumabas pati yung true colors nya",
    ]
    ex_cols = st.columns(len(examples))
    for ec, ex in zip(ex_cols, examples):
        with ec:
            if st.button(f"{ex[:35]}…", key=f"ex_{ex[:10]}", use_container_width=True, help=ex):
                st.session_state["analyzer_input"] = ex
                st.session_state.analysis_result = None
                st.session_state.all_model_results = None

    st.markdown("")

    # ── Input + Model picker ─────────────────────────
    col1, col2 = st.columns([4,1])

    with col1:
        user_input = st.text_area(
            "Enter a Taglish sentence or phrase.",
            value=st.session_state.get("analyzer_input", ""),
            placeholder="e.g., 'Ang galing niya, suportahan natin siya!'",
            height=120,
            key="text_input",
        )
    with col2:
        with st.container(border=True):
            chosen_model = st.selectbox("Select Model", MODELS, index=2)
            analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
            clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state["analyzer_input"] = ""
        st.session_state.analysis_result = None
        st.session_state.all_model_results = None
        st.rerun()

   # ══════════════════════════════════════════════
    # SINGLE MODEL INFERENCE
    # ══════════════════════════════════════════════
    if analyze_btn and user_input.strip() and len(user_input.strip().split()) < 2:
        st.warning("⚠️ Try a full phrase or sentence — a single word doesn't give the model enough context.")
    elif analyze_btn and user_input.strip():
        with st.spinner(f"{chosen_model} is analyzing…"):
            pred_label, scores = predict(user_input, chosen_model)

        st.session_state.analysis_result = {
            "label": pred_label,
            "scores": scores,
            "model": chosen_model,
            "text": user_input
        }
    elif analyze_btn:
        st.warning("⚠️ Please enter a sentence before analyzing.")
    
    if st.session_state.analysis_result:

        pred_label = st.session_state.analysis_result["label"]
        scores = st.session_state.analysis_result["scores"]
        chosen_model = st.session_state.analysis_result["model"]
        user_input = st.session_state.analysis_result["text"]

        cls_map = {
            "Positive": ("positive","😊","#22c55e"),
            "Neutral":  ("neutral","😐","#7C7C7C"),
            "Negative": ("negative","😤","#E8111A")
        }

        pred_label = pred_label.capitalize()
        css_cls, emoji, color = cls_map[pred_label]
        conf = max(scores)

        st.markdown(f"""
        <div class="result-box {css_cls}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{pred_label}</div>
            <div class="result-conf">
                Confidence: <strong>{conf:.1%}</strong> | Model: <strong>{chosen_model}</strong>
            </div>
            <div style="margin-top:.6rem;font-size:.8rem;color:#64748b;font-style:italic;">
                "{user_input}"
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Score Breakdown:**")

        cols = st.columns(3)
        class_names = ["Negative","Neutral","Positive"]
        colors = ["#e8111a","#7C7C7C","#22c55e"]

        for col, name, c, score in zip(cols, class_names, colors, scores):
            with col:
                is_winner = name == pred_label
                border = f"2px solid {c}" if is_winner else "1px solid #E5E7EB"
                weight = "800" if is_winner else "600"
                st.markdown(f"""
                <div style="border:{border};border-radius:12px;padding:14px;background:white;">
                    <div style="font-weight:{weight};text-align:center;margin-bottom:6px;">{name}{' ✓' if is_winner else ''}</div>
                    <div style="text-align:center;font-size:13px;margin-top:6px;color:#6B7280;">
                        {score:.2%}
                    </div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style="width:{score*100}%; background:{c};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-box" style="background:#F8FAFC;border:2px dashed #CBD5E1;">
            <div class="result-emoji" style="opacity:.4;">—</div>
            <div class="result-label" style="color:#94a3b8;">No analysis yet</div>
            <div class="result-conf" style="color:#94a3b8;">
                Enter a sentence above and click Analyze Sentiment to see a result here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Compare Across All Models</div>', unsafe_allow_html=True)


    run_all = st.button("Run on All Models", key="run_all_models")

    if run_all and not st.session_state.get("text_input", "").strip():
        st.warning("⚠️ Please enter a sentence before comparing models.")
    elif run_all:

        user_text = st.session_state["text_input"]

        all_preds = []

        for m in MODELS:

            lbl, sc = predict(user_text, m)

            all_preds.append({
                "Model": m,
                "Prediction": lbl,
                "Confidence": float(max(sc)),
                "Negative": float(sc[0]),
                "Neutral": float(sc[1]),
                "Positive": float(sc[2])
            })

        st.session_state.all_model_results = all_preds

    if st.session_state.all_model_results:

        df_all = pd.DataFrame(st.session_state.all_model_results)

        fig_all = go.Figure()

        for model in df_all["Model"]:
            row = df_all[df_all["Model"] == model].iloc[0]

            fig_all.add_trace(go.Bar(
            name=model,
            x=["Negative","Neutral","Positive"],
            y=[row["Negative"], row["Neutral"], row["Positive"]],
            marker_color=MODEL_COLORS[model], 
            text=[f"{v:.2f}" for v in [row["Negative"],row["Neutral"],row["Positive"]]],
            textposition="outside",
            textfont=dict(color="#1e293b", size=12, family="Montserrat"),
            cliponaxis=False,
        ))

        fig_all.update_layout(
            barmode="group",
            yaxis=dict(range=[0,1.15]), 
            title="Sentiment Distribution per Model",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )

        st.plotly_chart(fig_all, use_container_width=True)

        r1, r2, r3, r4 = st.columns(4)

        for rc, row in zip([r1,r2,r3,r4], st.session_state.all_model_results):

            pred = row["Prediction"].capitalize()

            cls_c = {"Positive":"green","Neutral":"","Negative":"red"}[pred]

            with rc:
                st.markdown(f"""
                <div class="kpi-card {cls_c}" style="padding:.8rem;">
                    <div class="kpi-label">{row['Model']}</div>
                    <div class="kpi-value" style="font-size:1.1rem;">{pred}</div>
                    <div class="kpi-sub">{row['Confidence']:.1%} conf.</div>
                </div>
                """, unsafe_allow_html=True)


    st.markdown("""
    <div class="pbb-footer">
        The predictions are generated using trained and fine-tuned sentiment models.
    </div>
    """, unsafe_allow_html=True)
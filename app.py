"""Run: streamlit run app.py"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Recommender", page_icon="🌾", layout="wide")

EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘", "pigeonpeas": "🫛",
    "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘", "lentil": "🫘", "pomegranate": "🍎",
    "banana": "🍌", "mango": "🥭", "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈",
    "apple": "🍎", "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "☁️",
    "jute": "🌿", "coffee": "☕",
}


@st.cache_resource
def load():
    return joblib.load("model.pkl"), json.load(open("metrics.json"))


try:
    model, m = load()
except FileNotFoundError:
    st.error("Run `python train.py` first to create model.pkl and metrics.json.")
    st.stop()

F = m["features"]
R = m["ranges"]

st.title("🌾 Smart Crop Recommendation System")
st.caption(f"Random Forest classifier · {len(m['classes'])} crops · test accuracy {m['accuracy']:.1%}")

# ---------- Sidebar inputs ----------
st.sidebar.header("Field conditions")
labels = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)",
}
vals = {}
for f in F:
    lo, hi, med = R[f]
    vals[f] = st.sidebar.slider(labels[f], float(lo), float(hi), float(med))
row = pd.DataFrame([vals])[F]

tab1, tab2, tab3 = st.tabs(["🌱 Recommend", "📊 Model insights", "📁 Batch upload"])

# ---------- Tab 1 ----------
with tab1:
    proba = model.predict_proba(row)[0]
    order = np.argsort(proba)[::-1][:3]
    top = model.classes_[order[0]]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Best crop")
        st.metric(f"{EMOJI.get(top, '🌱')} {top.title()}", f"{proba[order[0]]:.0%} confidence")
        if proba[order[0]] < 0.5:
            st.warning("Low confidence: conditions fit several crops.")
        st.write("**Runner-ups**")
        for i in order[1:]:
            st.write(f"{EMOJI.get(model.classes_[i], '🌱')} {model.classes_[i].title()}: {proba[i]:.0%}")
    with c2:
        st.subheader("Top 3 probabilities")
        st.bar_chart(pd.Series({model.classes_[i].title(): proba[i] for i in order}))

    st.divider()
    st.subheader("🔮 What-if: change one factor")
    feat = st.selectbox("Factor to vary", F, index=F.index("rainfall"), format_func=lambda x: labels[x])
    grid = np.linspace(R[feat][0], R[feat][1], 40)
    sweep = pd.DataFrame([{**vals, feat: g} for g in grid])[F]
    sp = model.predict_proba(sweep)
    chart = pd.DataFrame({model.classes_[i].title(): sp[:, i] for i in order}, index=np.round(grid, 1))
    chart.index.name = labels[feat]
    st.line_chart(chart)
    st.caption("Shows how the top 3 crops' probabilities shift as this one factor changes.")

# ---------- Tab 2 ----------
with tab2:
    a, b = st.columns(2)
    with a:
        st.subheader("Feature importance")
        imp = pd.Series(m["importances"]).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(imp.index, imp.values, color="#2e7d32")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    with b:
        st.subheader("Confusion matrix")
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(m["classes"])))
        ax.set_yticks(range(len(m["classes"])))
        ax.set_xticklabels(m["classes"], rotation=90, fontsize=6)
        ax.set_yticklabels(m["classes"], fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    st.write("**Best hyperparameters (GridSearchCV):**", m["best_params"])

# ---------- Tab 3 ----------
with tab3:
    st.write(f"Upload a CSV with columns: `{', '.join(F)}`")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        d = pd.read_csv(up)
        if not set(F).issubset(d.columns):
            st.error(f"Missing columns: {sorted(set(F) - set(d.columns))}")
        else:
            d["recommended_crop"] = model.predict(d[F])
            d["confidence"] = (model.predict_proba(d[F]).max(axis=1) * 100).round(1)
            st.dataframe(d)
            st.download_button("Download results", d.to_csv(index=False), "crop_predictions.csv")

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from modeling.hybrid_model import process_conversation

st.set_page_config(page_title="Emotion Drift AI", layout="wide")

st.title("🧬 Emotion Drift Analyzer (LSTM + Ollama)")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# ---------------- INPUT ----------------
col1, col2 = st.columns([1, 4])

with col1:
    user = st.selectbox("Speaker", ["A", "B"])

with col2:
    msg = st.text_input("Enter message")

if st.button("➕ Add Message"):
    if msg:
        st.session_state.messages.append(f"{user}: {msg}")

# ---------------- DISPLAY CONVERSATION ----------------
st.subheader("💬 Conversation")

if len(st.session_state.messages) == 0:
    st.info("No messages yet. Add at least 3 messages.")
else:
    for i, m in enumerate(st.session_state.messages):
        st.markdown(f"**{i+1}. {m}**")

# ---------------- ANALYZE BUTTON ----------------
if len(st.session_state.messages) >= 3:
    if st.button("🚀 Analyze Conversation"):
        st.session_state.run_analysis = True

# ---------------- ANALYSIS ----------------
if st.session_state.run_analysis:

    result = process_conversation(st.session_state.messages)

    if result:

        vectors = result["vectors"]
        risk = result["risk"]

        # ================= GRAPH =================
        st.subheader("📊 Emotion Drift")

        fig, ax = plt.subplots(figsize=(6, 3))

        emotion_names = ["Joy", "Anger", "Sadness", "Fear", "Neutral"]

        for i, name in enumerate(emotion_names):
            values = [v[i] for v in vectors]
            ax.plot(values, label=name)

        ax.legend(loc="upper right")
        ax.set_title("Emotion Drift")

        plt.tight_layout()
        st.pyplot(fig)

        # ================= HEATMAP =================
        st.subheader("🔥 Emotion Heatmap")

        heatmap = np.array(vectors).T

        fig2, ax2 = plt.subplots(figsize=(6, 2))
        im = ax2.imshow(heatmap, aspect='auto')

        ax2.set_yticks(range(len(emotion_names)))
        ax2.set_yticklabels(emotion_names)

        plt.colorbar(im)
        plt.tight_layout()
        st.pyplot(fig2)

        # ================= RISK =================
        st.subheader("⚠️ Risk Score (LSTM)")

        if risk > 0.7:
            st.error(f"High Risk: {risk:.2f}")
        elif risk > 0.4:
            st.warning(f"Medium Risk: {risk:.2f}")
        else:
            st.success(f"Low Risk: {risk:.2f}")

        st.info("ℹ️ LSTM shows current risk, Ollama explains overall behavior.")

        # ================= EXPLANATION =================
        st.subheader("🧠 AI Explanation")
        st.write(result["explanation"])

        # ================= DOWNLOAD =================
        st.subheader("📄 Download Report")

        df = pd.DataFrame({
            "message": st.session_state.messages,
            "risk": [risk] * len(st.session_state.messages)
        })

        csv = df.to_csv(index=False).encode()

        st.download_button(
            "⬇️ Download CSV Report",
            csv,
            "emotion_analysis_report.csv"
        )

    # Reset flag after run
    st.session_state.run_analysis = False

# ---------------- RESET ----------------
if st.button("🔄 Reset Conversation"):
    st.session_state.messages = []
    st.session_state.run_analysis = False
import streamlit as st
import os
from plag_predictor import read_file, tokenize, score_single_file

# Page Config
st.set_page_config(page_title="Plagiarism Risk Predictor", page_icon="🔍", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4caf50, #f44336);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Assignment Plagiarism Risk Predictor")
st.write("Upload a file (.txt, .py, .md) to analyze its stylistic patterns and plagiarism risk.")

# File Uploader
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'py', 'md', 'html'])

if uploaded_file is not None:
    # Save temporarily to use your existing read_file logic or read directly
    bytes_data = uploaded_file.getvalue()
    text = bytes_data.decode("utf-8", errors="ignore")
    
    with st.spinner('Analyzing linguistic patterns...'):
        tokens = tokenize(text)
        
        if len(tokens) < 10:
            st.error("The file is too short for a reliable analysis. Please upload a longer text.")
        else:
            # Get Scores from your plag_predictor.py
            results = score_single_file(text, tokens)
            risk_score = results['overall_risk_score']
            
            # --- Visualizing Results ---
            st.divider()
            
            # Risk Gauge
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Overall Risk", value=f"{risk_score * 100:.1f}%")
                if risk_score < 0.3:
                    st.success("✅ Low Risk")
                elif risk_score < 0.6:
                    st.warning("🟡 Moderate Risk")
                else:
                    st.error("🔴 High Risk")
            
            with col2:
                st.write("**Risk Analysis**")
                st.progress(risk_score)
                st.caption("Score based on vocabulary richness, passive voice, and structural uniformity.")

            st.subheader("📊 Feature Breakdown")
            
            # Create a grid for the specific metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Vocab Richness", f"{results['vocabulary_richness']*100:.1f}%", help="Higher is better (more original)")
            m2.metric("Phrase Density", f"{results['repeated_phrase_density']*100:.1f}%", delta_color="inverse")
            m3.metric("Style Shifts", f"{results['style_shift_score']}", help="Abrupt changes in writing complexity")
            
            m4, m5, m6 = st.columns(3)
            m4.metric("Uniformity", f"{results['structural_uniformity']*100:.1f}%", help="High uniformity suggests templates")
            m5.metric("Passive Voice", f"{results['passive_voice_ratio']*100:.1f}%")
            m6.metric("Formality", f"{results['formality_score']*100:.1f}%")

            # Show raw text option
            with st.expander("View Analyzed Text"):
                st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

else:
    st.info("Please upload a file to begin the analysis.")

# Footer
st.sidebar.markdown("### How it works")
st.sidebar.info("""
This tool uses **TF-IDF** and **Heuristic AI scoring**. 
It doesn't just look for matches online; it analyzes the *DNA* of the writing to see if it looks like a human wrote it or if it was pieced together.
""")
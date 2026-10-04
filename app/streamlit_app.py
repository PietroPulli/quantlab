"""Quantlab web app: pick an idea and an asset, get an honest verdict against buy & hold.

Run from the project root with:  streamlit run app/streamlit_app.py

Two views on the same truth machine:
- simple.py:   three plain choices, one plain answer (the default);
- advanced.py: every parameter, custom rules and all the statistics.
This file only draws the frame. Every number comes from the quantlab library.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the views can `import common`

import advanced  # noqa: E402
import simple  # noqa: E402
from common import CSS  # noqa: E402

st.set_page_config(page_title="Quantlab", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.title("Quantlab")
st.caption(
    "Prende un'idea di investimento, la prova sul passato con costi reali e ti dice onestamente "
    "se avrebbe battuto il semplice comprare e tenere. Non è un consiglio di investimento."
)
view = st.radio("Vista", ["Semplice", "Approfondita"], key="view", horizontal=True, label_visibility="collapsed")

if view == "Semplice":
    simple.render()
else:
    advanced.render()

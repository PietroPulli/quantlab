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

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))  # so the views can `import common`
# Use the library straight from this repository's src/ folder. A hosting service may keep an
# old installed copy of quantlab after a git push; this way the app and the library always match.
sys.path.insert(0, str(APP_DIR.parent / "src"))

import advanced  # noqa: E402
import paper_view  # noqa: E402
import scanner_view  # noqa: E402
import simple  # noqa: E402
from common import CSS  # noqa: E402

st.set_page_config(page_title="Quantlab", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="ql-brand"><b>Quantlab</b><span>Verifica storica di strategie di investimento: '
    "costi reali, dati mai visti, controllo della fortuna. Non è un consiglio di investimento.</span></div>",
    unsafe_allow_html=True,
)
VIEWS = {"Semplice": simple.render, "Approfondita": advanced.render, "Scanner": scanner_view.render,
         "Conto demo": paper_view.render}
view = st.radio("Vista", list(VIEWS), key="view", horizontal=True, label_visibility="collapsed")
VIEWS[view]()

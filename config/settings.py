"""
Application settings and Streamlit page configuration.
"""

import streamlit as st


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Algorithms Comparison",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# This is incomplete

import tensorflow as tf
import streamlit as st
import warnings
import time

warnings.filterwarnings("ignore")



# ---------------------------------------FUNCTIONS------------------------------------------------------------------------------
def stream_data(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def get_latest_bitcoin_data() -> None:
    pass


def plot_data() -> None:
    pass


def get_predictions() -> list:
    pass





# ----------------------------------------STREAMLIT PAGE----------------------------------------------------------------------------
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        /* Make the main container wider */
        .main-container {
            max-width: 95% !important;
            margin: auto;
        }

        /* Center all content inside containers */
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            width: 100%;s
        }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Bitcoin Analyzer 📈📊")
st.header("A Model to predict the price of Bitcoin using data in the wild that updates everyday")

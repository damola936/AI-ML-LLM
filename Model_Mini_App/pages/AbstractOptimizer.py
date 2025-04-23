import tensorflow as tf
import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.edge.service import Service
# from selenium.webdriver.edge.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.microsoft import EdgeChromiumDriverManager
import streamlit as st
import time
import requests
from bs4 import BeautifulSoup
import tensorflow_hub as hub
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np


# FUNCTIONS............

def stream_data(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def get_pubmed_abstract(link: str) -> dict | None:
    """Scrapes title, abstract, and copyright information from a PubMed link using BeautifulSoup."""

    result = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    try:
        # Send an HTTP request to fetch the webpage content
        response = requests.get(link, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 404, 500)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_element = soup.select_one("#full-view-heading h1")
        result["title"] = title_element.text.strip() if title_element else "Title not found"

        # Extract abstract
        abstract_element = soup.select_one("#eng-abstract")
        result["abstract"] = abstract_element.text.strip() if abstract_element else "Abstract not found"

        # Extract copyright info
        copyright_element = soup.select_one("#copyright")
        result["copyright"] = copyright_element.text.strip() if copyright_element else "Copyright info not found"

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve data: {e}")
        return None  # Return None if there's a major issue

    return result


# FOR SELENIUM USAGE WITH WINDOW OPENED
# def get_pubmed_abstract(link: str) -> dict | None:
#     """Scrapes title, abstract, and copyright information from a PubMed link."""
#
#     result = {}
#
#     # Setup Edge options for headless execution
#     options = Options()
#     # options.add_argument("--headless")  # Run in headless mode
#     options.add_argument("--disable-gpu")  # Fix potential rendering issues
#     options.add_argument("--log-level=3")  # Suppress logs
#     options.add_argument("--window-size=1920,1080")  # Ensure correct element rendering
#     options.add_argument("--disable-blink-features=AutomationControlled")  # Bypass bot detection
#     options.add_experimental_option("excludeSwitches", ["enable-automation"])  # Hide automation control
#     options.add_experimental_option("useAutomationExtension", False)
#
#     # Initialize Edge WebDriver
#     service = Service(EdgeChromiumDriverManager().install())
#     driver = webdriver.Edge(service=service, options=options)
#
#     try:
#         driver.get(link)  # Open the page
#
#         # Wait for the title to load
#         try:
#             title_element = WebDriverWait(driver, 5).until(
#                 EC.presence_of_element_located((By.XPATH, '//*[@id="full-view-heading"]/h1'))
#             )
#             result["title"] = title_element.text
#         except Exception as e:
#             print(f"Error retrieving title: {e}")
#             result["title"] = "Title not found"
#
#         # Wait for the abstract
#         try:
#             abstract_element = WebDriverWait(driver, 1).until(
#                 EC.presence_of_element_located((By.XPATH, '//*[@id="eng-abstract"]'))
#             )
#             result["abstract"] = abstract_element.text
#         except Exception as e:
#             print(f"Error retrieving abstract: {e}")
#             result["abstract"] = "Abstract not found"
#
#         # Wait for copyright info
#         try:
#             copyright_element = WebDriverWait(driver, 1).until(
#                 EC.presence_of_element_located((By.XPATH, '//*[@id="copyright"]'))
#             )
#             result["copyright"] = copyright_element.text
#         except Exception as e:
#             print(f"Error retrieving copyright info: {e}")
#             result["copyright"] = "Copyright info not found"
#
#     except Exception as e:
#         print(f"Failed to retrieve data: {e}")
#         return None  # Return None if there's a major issue
#
#     finally:
#         driver.quit()  # Ensure the browser closes
#
#     return result


def process_abstract(abstract_text: str):
    abstract_list = []
    split_abstract = abstract_text.split(".")

    def split_characters(text):
        return " ".join(list(text))

    with st.spinner("Processing abstract..."):
        for i, s in enumerate(split_abstract):
            abstract_dict = {
                "line_number": i,
                "text": s,
                "total_lines": len(split_abstract)
            }
            abstract_list.append(abstract_dict)  # construct our target dictionary list to be turned to a dataframe

        abstract_df = pd.DataFrame(abstract_list)
        print(abstract_df.head())
        abstract_sentences = abstract_df["text"].to_list()
        abstract_chars = [split_characters(s) for s in abstract_sentences]
        abstract_line_numbers_one_hot = tf.one_hot(abstract_df["line_number"].to_numpy(),
                                                   depth=15)  # one hot encode our line numbers
        abstract_total_lines_one_hot = tf.one_hot(abstract_df["total_lines"].to_numpy(),
                                                  depth=20)  # one hot encode our total lines
        abstract_tribrid_text = tf.data.Dataset.from_tensor_slices(
            (abstract_line_numbers_one_hot, abstract_total_lines_one_hot, abstract_sentences, abstract_chars)
        ).batch(32).prefetch(tf.data.AUTOTUNE)
    return abstract_tribrid_text



# Prepare inputs as a dictionary matching the input names used during training
def prepare_inputs_for_prediction(dataset):
    inputs = []
    for batch in dataset:
        inputs.extend(list(zip(*batch)))  # flatten batches

    input_tensors = {
        "line_number_inputs": tf.convert_to_tensor([x[0] for x in inputs], dtype=tf.float32),
        "total_lines_inputs": tf.convert_to_tensor([x[1] for x in inputs], dtype=tf.float32),
        "token_inputs": tf.convert_to_tensor([x[2] for x in inputs], dtype=tf.string),
        "char_inputs": tf.convert_to_tensor([x[3] for x in inputs], dtype=tf.string)
    }
    return input_tensors



def abstract_optimizer_predict(dataset):
    """
    Makes Predictions using dataset
    Args:
        dataset: The Prefetched Dataset

    Returns: Predictions on the Dataset

    """
    # Load Model
    with st.spinner("Optimizing Abstract..."):
        with st.spinner("Loading Model..."):
            model = tf.keras.models.load_model(
                os.path.abspath("models/AbstractOptimizer20K.keras")
            )
        prepared_inputs = prepare_inputs_for_prediction(dataset)
        pred_probs = model.predict(prepared_inputs)
        print(pred_probs[:10])
        predictions = tf.argmax(pred_probs, axis=1)
        return predictions



def organize_predictions(predictions:list, data:str) -> None:
    """Organizes and writes data to the streamlit app based on predictions"""
    data_list = data.split(".")
    class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    predictions_map = [class_names[p] for p in predictions]
    full_data = list(zip(predictions_map, data_list))
    backgrounds = []
    conclusions = []
    methods = []
    objectives = []
    results = []
    for d in full_data:
        if d[0] == "BACKGROUND":
            backgrounds.append(d[1])
        elif d[0] == "CONCLUSIONS":
            conclusions.append(d[1])
        elif d[0] == "METHODS":
            methods.append(d[1])
        elif d[0] == "OBJECTIVE":
            objectives.append(d[1])
        else:
            results.append(d[1])
    
    if len(backgrounds) > 0:
        st.subheader("Background")
        for text in backgrounds:
            st.write_stream(stream_data(text))
    if len(objectives) > 0:
        st.subheader("Objectives")
        for text in objectives:
            st.write_stream(stream_data(text))
    if len(methods) > 0:
        st.subheader("Methods")
        for text in methods:
            st.write_stream(stream_data(text))
    if len(results) > 0:
        st.subheader("Results")
        for text in results:
            st.write_stream(stream_data(text))
    if len(conclusions) > 0:
        st.subheader("Conclusions")
        for text in conclusions:
            st.write_stream(stream_data(text))




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
            width: 100%;
        }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Abstract Optimizer 📃📎")
st.header("A PubMed Abstract Optimizer Made using NLP")
st.write("With all due credit to the research paper")
st.divider()

st.subheader("Model Building")
st.write("The model was made using the PubMed200K RCT Dataset 20K  and 200K versions")
st.write(
    "The Model was made using an RNN(Recurrent Neural Network). Specifically a Pretrained Token Embedding,"
    "Character embeddings and Positional embeddings. The Model architecture is shown below."
)
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Center column
    st.image("content/architecture.png", caption="Model Architecture", width=800)  #TODO:Remove Background from image

st.divider()

st.subheader("What the Model & App does")
st.write(
    "The App takes in your link to a PubMed research paper, gets the abstract from that paper. And organizes it for better understanding")
st.divider()

st.subheader("Try it out yourself")

with st.container():
    is_finished = False

    st.markdown('<div class="centered">', unsafe_allow_html=True)  # START DIV BLOCK

    ph1, x, ph2 = st.columns([1, 2, 1])
    with x:
        with st.chat_message("ai"):
            st.write("Paste a link to a PubMed RCT research here.")
            prompt = st.chat_input("Post your link here")
            if prompt:
                if "pubmed" and "gov" not in prompt:
                    st.toast("❌ Not a PubMed Link", icon="⚠️")
                else:
                    st.toast("✔️ Success", icon="🎉")
                    is_finished = True

    st.markdown('</div>', unsafe_allow_html=True)  # CLOSE DIV BLOCK

    if not is_finished:
        pass
    else:
        with st.container():
            show_chat_input = False
            col1, col2 = st.columns([1, 1])

            with col1:
                with st.container():  # Keep this inside the container to ensure Streamlit layout works correctly

                    if is_finished:
                        with st.spinner():
                            st.markdown('<h2 style="color:lightblue;">Un-Optimized Abstract</h2>',
                                        unsafe_allow_html=True)
                            st.divider()
                            abstract = get_pubmed_abstract(prompt)

                            st.subheader("Title")
                            st.write_stream(stream_data(abstract["title"]))

                            st.subheader("Abstract")
                            st.write_stream(stream_data(abstract["abstract"]))

                            st.write_stream(stream_data(abstract["copyright"]))

                            is_finished = False

            with col2:
                with st.container():  # Ensure Streamlit containerization

                    if abstract:
                        st.markdown("""
                            <h2 style="
                                background: linear-gradient(to right, red, orange, yellow, green, cyan, blue, violet);
                                -webkit-background-clip: text;
                                color: transparent;
                                font-weight: bold;
                                text-align: center;">
                                Optimized Abstract
                            </h2>
                        """, unsafe_allow_html=True)
                        st.divider()
                        tribrid_dataset = process_abstract(abstract["abstract"])
                        optimizer_predictions = list(abstract_optimizer_predict(tribrid_dataset))
                        organize_predictions(predictions=optimizer_predictions, data=abstract["abstract"])

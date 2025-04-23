import streamlit as st
import tensorflow as tf
import time
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import  Pipeline


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

st.title("Disaster Checker 💥💥")
st.header("A Disaster Checker App made using NLP")
st.divider()

st.write(
    "This is a disaster checker app made using NLP(Natural Language Processing) utilising a RNN architecture."
    "RNN Means Recurrent Neural Networks which learn from sequences of data. It is mostly utilised in understanding text,"
    " the RNN model I built was built was upon a pretrained encoder model called Universal Sentence encoder. USE(Universal Sentence Encoder)"
    "was built upon BERT model and performs with similar results in the same domain space."
)
st.divider()
st.write("In this app you will have access to 2 models. Our baseline model which is quick, does not utilize an RNN "
         "but a Machine Learning Model called Naive Bayes, specifically MultiNomial Naive Bayes")

st.divider()
st.divider()

st.subheader("Chat with the models below")
st.divider()


def load_and_classify(model_path:str, text:str) -> str:
    """
    Loads the Machine Learning Model, processes the text and classifies the text
    Args:
        text: The text to be processed
        model_path: The path to the Model

    Returns:
        Whether the text classifies as a disaster text or not
    """
    #TODO:Use Deep Learning Model
    return "text"


def stream_data(text:str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


with st.container():
    st.markdown('<div class="centered">', unsafe_allow_html=True)  # START DIV BLOCK
    with st.chat_message("ai"):

        st.write("Hello! I am a Deep Learning Model trained to classify disaster tweets. Try me out!")
        st.write("Enter a message or Tweet to test me out!")
        
        prompt = st.chat_input("Say something")
        if prompt:
            with st.status("Assessing Message", state="running", expanded=True) as status:
                try:
                    result = load_and_classify("models/ML.pkl", prompt)
                except Exception as e:
                    st.error(e)
                else:
                    status.update("Complete", state="complete", expanded=False)
                    result_text = f"The Model classified your text as :blue[{result}]"
                    st.write_stream(stream_data(result_text))


    st.markdown('</div>', unsafe_allow_html=True)  # CLOSE DIV BLOCK

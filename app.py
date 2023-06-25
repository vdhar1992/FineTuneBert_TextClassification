import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from preprocess import *


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('vdhar1992/TxtClf_FineTuneBERT')
    return tokenizer,model



tokenizer,model = get_model()

user_input = st.text_area('Enter tweet to classify')
button = st.button("Classify")


d= {
    1:"Disaster",
    0:"Non Disaster"
}

if user_input and button :
   
    test_sample = preprocess_text(user_input)
    test_sample = tokenizer([test_sample], padding=True, truncation=True, max_length=60,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])
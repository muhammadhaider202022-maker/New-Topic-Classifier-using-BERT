import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import pandas as pd

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizerFast.from_pretrained("saved_model")

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("News Topic Classifier")
st.write("Enter a news headline below:")

headline = st.text_input("News Headline")

if st.button("Classify"):
    if headline.strip():
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = torch.argmax(outputs.logits, dim=1).item()
        st.success(f"Predicted Topic: **{labels[pred]}**")
        
        # Show probability bar chart
        df = pd.DataFrame({
            "Category": labels,
            "Probability (%)": (probs * 100).round(2)
        })
        st.bar_chart(df.set_index("Category"))
    else:
        st.warning("Please enter a headline.")